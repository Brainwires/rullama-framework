/**
 * Shell-command risk classification for `execute_command`.
 *
 * {@link classifyCommand} parses a command the way a POSIX shell splits it
 * (quotes, escapes, `;` / `&&` / `||` / `|` / `&` / newlines, redirects,
 * `$(…)` / backtick / `<(…)` substitutions, `sh -c` / `eval` scripts) and
 * checks each simple command's real argv, after stripping wrappers such as
 * `sudo`, `env`, `nohup`, `time`, `nice`, `timeout` and `xargs`.
 *
 * Quoting rule: text quoted as an argument to an ordinary command is data,
 * not code, so `echo "rm -rf /"` and `git commit -m "rm -rf /"` are `ok`. A
 * quoted script handed to a shell (`bash -c "…"`, `eval "…"`) and every
 * command substitution (`echo "$(rm -rf /)"`) is parsed and classified as
 * code. Two checks deliberately look at the raw text, quotes included: fork
 * bombs (block) and SQL `DROP` / `TRUNCATE` (warn), since SQL nearly always
 * reaches its client inside a quoted argument.
 *
 * This is a guard rail against accidents, not a sandbox: variables, globs,
 * aliases and indirection (`$(echo rm) -rf /`) are not evaluated.
 *
 * @module
 */

/** How risky a command is: `block` refuses it, `warn` deserves a confirmation. */
export type CommandVerdict = "block" | "warn" | "ok";

/** Result of {@link classifyCommand}. */
export interface CommandClassification {
  /** `block` when any block-level pattern matched, else `warn` when any
   * warn-level one did, else `ok`. */
  verdict: CommandVerdict;
  /** One human-readable reason per match, block-level first, deduplicated.
   * Empty when `verdict` is `ok`. */
  reasons: string[];
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

interface Redirect {
  op: string;
  target: string;
}

interface SimpleCommand {
  argv: string[];
  redirects: Redirect[];
}

interface Parsed {
  /** Pipelines, each a list of simple commands joined by `|`. */
  pipelines: SimpleCommand[][];
  /** Bodies of `$(…)`, backtick, `<(…)` and `>(…)` substitutions. */
  substitutions: string[];
}

const REDIRECT_OPS = [
  "&>>",
  "&>",
  ">>",
  ">|",
  ">&",
  ">",
  "<<<",
  "<<",
  "<&",
  "<>",
  "<",
];

/** Index of the `)` closing a group whose body starts at `start` (quotes
 * respected), or `s.length` when it is unterminated. */
function closingParen(s: string, start: number): number {
  let depth = 1;
  for (let j = start; j < s.length; j++) {
    const c = s[j];
    if (c === "'" || c === '"') {
      j = closingQuote(s, j);
      continue;
    }
    depth += PAREN_DELTA[c] ?? 0;
    if (depth === 0) return j;
  }
  return s.length;
}

const PAREN_DELTA: Record<string, number> = { "(": 1, ")": -1 };

/** Index of the quote matching the one at `j`, or `s.length`. */
function closingQuote(s: string, j: number): number {
  const end = s.indexOf(s[j], j + 1);
  return end === -1 ? s.length : end;
}

/** A small shell lexer: enough of POSIX word splitting to see real argv. */
class Lexer {
  private i = 0;
  private word = "";
  private inWord = false;
  private pendingRedirect: string | null = null;
  private cmd: SimpleCommand = { argv: [], redirects: [] };
  private pipeline: SimpleCommand[] = [];
  readonly out: Parsed = { pipelines: [], substitutions: [] };

  constructor(private readonly s: string) {}

  parse(): Parsed {
    while (this.i < this.s.length) this.step();
    this.endPipeline();
    return this.out;
  }

  private step(): void {
    if (this.quotingOrExpansion()) return;
    const c = this.s[this.i];
    if (c === "#" && !this.inWord) return this.comment();
    if (this.operator()) return;
    this.i++;
    if (c === " " || c === "\t") this.endWord();
    else this.append(c);
  }

  /** Quotes, escapes and `$…` / `` ` `` / `<(` / `>(` expansions. */
  private quotingOrExpansion(): boolean {
    const c = this.s[this.i];
    switch (c) {
      case "'":
        this.singleQuoted();
        return true;
      case '"':
        this.doubleQuoted();
        return true;
      case "\\":
        this.escaped();
        return true;
      case "`":
        this.append(this.backtick());
        return true;
      case "$":
        return this.dollar();
      case "<":
      case ">":
        return this.processSubstitution(c);
    }
    return false;
  }

  /** `$(…)` or `${…}` at `this.i`; plain `$` is left to the caller. */
  private dollar(): boolean {
    const next = this.s[this.i + 1];
    if (next === "(") {
      this.i++;
      this.append(this.substitution("$"));
      return true;
    }
    if (next === "{") {
      this.braceVar();
      return true;
    }
    return false;
  }

  private processSubstitution(sigil: string): boolean {
    if (this.s[this.i + 1] !== "(") return false;
    this.i++;
    this.append(this.substitution(sigil));
    return true;
  }

  private append(text: string): void {
    this.word += text;
    this.inWord = true;
  }

  private singleQuoted(): void {
    const end = this.s.indexOf("'", this.i + 1);
    const stop = end === -1 ? this.s.length : end;
    this.append(this.s.slice(this.i + 1, stop));
    this.i = stop + 1;
  }

  private doubleQuoted(): void {
    this.i++;
    this.inWord = true;
    while (this.i < this.s.length && this.s[this.i] !== '"') {
      this.doubleQuotedChar();
    }
    this.i++;
  }

  private doubleQuotedChar(): void {
    const c = this.s[this.i];
    const next = this.s[this.i + 1];
    if (c === "\\" && next !== undefined && '"\\$`\n'.includes(next)) {
      this.i += 2;
      if (next !== "\n") this.append(next);
      return;
    }
    if (c === "$" && this.dollar()) return;
    if (c === "`") return this.append(this.backtick());
    this.i++;
    this.append(c);
  }

  private escaped(): void {
    const next = this.s[this.i + 1];
    this.i += 2;
    if (next === undefined || next === "\n") return;
    this.append(next);
  }

  private braceVar(): void {
    const end = this.s.indexOf("}", this.i);
    const stop = end === -1 ? this.s.length : end + 1;
    this.append(this.s.slice(this.i, stop));
    this.i = stop;
  }

  private comment(): void {
    const end = this.s.indexOf("\n", this.i);
    this.i = end === -1 ? this.s.length : end;
  }

  /** `$(…)`, `<(…)`, `>(…)` with `this.i` on the `(`: record the body,
   * return the text to keep in the word. */
  private substitution(sigil: string): string {
    const start = this.i + 1;
    const close = closingParen(this.s, start);
    const body = this.s.slice(start, close);
    this.out.substitutions.push(body);
    this.i = Math.min(close + 1, this.s.length);
    return `${sigil}(${body})`;
  }

  private backtick(): string {
    const end = this.s.indexOf("`", this.i + 1);
    const stop = end === -1 ? this.s.length : end;
    const body = this.s.slice(this.i + 1, stop);
    this.out.substitutions.push(body);
    this.i = stop + 1;
    return `$(${body})`;
  }

  /** Command separators, pipes, subshell parens and redirects. */
  private operator(): boolean {
    const rest = this.s.slice(this.i, this.i + 3);
    for (const sep of ["&&", "||", ";;", "|&"]) {
      if (rest.startsWith(sep)) {
        this.i += 2;
        if (sep === "|&") this.endCommand();
        else this.endPipeline();
        return true;
      }
    }
    if (this.redirect(rest)) return true;
    const c = this.s[this.i];
    if (c === "|") {
      this.i++;
      this.endCommand();
      return true;
    }
    if (";&\n\r()".includes(c)) {
      this.i++;
      this.endPipeline();
      return true;
    }
    return false;
  }

  private redirect(rest: string): boolean {
    const op = REDIRECT_OPS.find((o) => rest.startsWith(o));
    if (!op) return false;
    // `2>file`: the digits are the fd, not a word.
    if (/^\d+$/.test(this.word)) {
      this.word = "";
      this.inWord = false;
    } else {
      this.endWord();
    }
    this.i += op.length;
    this.pendingRedirect = op;
    return true;
  }

  private endWord(): void {
    if (!this.inWord) return;
    if (this.pendingRedirect) {
      this.cmd.redirects.push({ op: this.pendingRedirect, target: this.word });
      this.pendingRedirect = null;
    } else {
      this.cmd.argv.push(this.word);
    }
    this.word = "";
    this.inWord = false;
  }

  private endCommand(): void {
    this.endWord();
    this.pendingRedirect = null;
    if (this.cmd.argv.length > 0 || this.cmd.redirects.length > 0) {
      this.pipeline.push(this.cmd);
    }
    this.cmd = { argv: [], redirects: [] };
  }

  private endPipeline(): void {
    this.endCommand();
    if (this.pipeline.length > 0) this.out.pipelines.push(this.pipeline);
    this.pipeline = [];
  }
}

// ---------------------------------------------------------------------------
// Wrapper stripping
// ---------------------------------------------------------------------------

interface Effective {
  /** Basename of the command actually run (`/bin/rm` → `rm`). */
  name: string;
  args: string[];
  /** Wrappers stripped on the way (`sudo`, `xargs`, …). */
  wrappers: string[];
}

/** Wrapper → its options that take a separate argument. */
const WRAPPERS: Record<string, string[]> = {
  sudo: ["-u", "-g", "-p", "-C", "-D", "-h", "-r", "-t", "-U", "-T"],
  doas: ["-u", "-C"],
  env: ["-u", "-C", "-S"],
  nohup: [],
  time: ["-f", "-o"],
  nice: ["-n"],
  ionice: ["-c", "-n", "-p"],
  command: [],
  builtin: [],
  exec: ["-a"],
  stdbuf: ["-i", "-o", "-e"],
  caffeinate: ["-t", "-w"],
  xargs: ["-I", "-L", "-n", "-P", "-d", "-E", "-s", "-a"],
  timeout: ["-s", "-k"],
};

const SHELL_KEYWORDS = new Set([
  "if",
  "then",
  "else",
  "elif",
  "do",
  "while",
  "until",
  "!",
  "{",
  "}",
]);

function basename(word: string): string {
  const slash = word.lastIndexOf("/");
  return slash === -1 ? word : word.slice(slash + 1);
}

/** Skip a wrapper's options (and `timeout`'s duration); returns the index
 * of the wrapped command. */
function skipWrapperOptions(
  argv: string[],
  i: number,
  wrapper: string,
): number {
  const withArg = WRAPPERS[wrapper];
  let j = i + 1;
  while (j < argv.length && argv[j].startsWith("-") && argv[j] !== "-") {
    if (argv[j] === "--") return j + 1;
    j += withArg.includes(argv[j]) ? 2 : 1;
  }
  // `timeout DURATION cmd`; `env`'s NAME=value pairs are skipped by the caller.
  return wrapper === "timeout" ? j + 1 : j;
}

/** Strip assignments, shell keywords and wrappers to find the real command. */
function effective(argv: string[]): Effective | null {
  const wrappers: string[] = [];
  let i = 0;
  while (i < argv.length) {
    const word = argv[i];
    if (/^[A-Za-z_]\w*=/.test(word) || SHELL_KEYWORDS.has(word)) {
      i++;
      continue;
    }
    const name = basename(word);
    if (!Object.hasOwn(WRAPPERS, name)) {
      return { name, args: argv.slice(i + 1), wrappers };
    }
    wrappers.push(name);
    i = skipWrapperOptions(argv, i, name);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/** System directories whose recursive removal / re-permissioning is fatal. */
const SYSTEM_ROOTS = new Set([
  "/bin",
  "/boot",
  "/dev",
  "/etc",
  "/home",
  "/lib",
  "/lib64",
  "/opt",
  "/proc",
  "/root",
  "/sbin",
  "/srv",
  "/sys",
  "/usr",
  "/var",
  "/Applications",
  "/Library",
  "/System",
  "/Users",
  "/Volumes",
  "/private",
]);

/** `${HOME}` → `$HOME`, collapse `//`, drop trailing `/` and `/.`. */
function normalizePath(target: string): string {
  let p = target.replace(/\$\{HOME\}/g, "$HOME").replace(/\/{2,}/g, "/");
  while (p.length > 1 && (p.endsWith("/") || p.endsWith("/."))) {
    p = p.endsWith("/.") ? p.slice(0, -2) || "/" : p.slice(0, -1);
  }
  return p;
}

/** `/`, `~`, `$HOME`, `*`, a system root, or any of those followed by `/*`. */
function isCriticalPath(target: string): boolean {
  const p = normalizePath(target);
  if (p === "*" || p === "./*") return true;
  const base = p.endsWith("/*") ? p.slice(0, -2) || "/" : p;
  return base === "/" || base === "~" || base === "$HOME" ||
    SYSTEM_ROOTS.has(base);
}

const RAW_DISK =
  /^\/dev\/(r?disk\d|sd[a-z]|hd[a-z]|vd[a-z]|xvd[a-z]|nvme\d|mmcblk\d|md\d|dm-\d|mapper\/|loop\d|nbd\d|sr\d)/;

const HARMLESS_DEVICE = /^\/dev\/(null|zero|u?random|stdout|stderr|tty|fd\/)/;

const SHELL_RC =
  /(^|\/)\.(bashrc|bash_profile|bash_login|bash_logout|profile|zshrc|zprofile|zshenv|zlogin|zlogout|kshrc|cshrc|tcshrc)$|(^|\/)\.config\/fish\/config\.fish$/;

const SHELLS = new Set(["sh", "bash", "zsh", "dash", "ksh", "fish", "ash"]);

/** Positional (non-option) arguments, honouring `--`. */
function operands(args: string[]): string[] {
  const out: string[] = [];
  let options = true;
  for (const a of args) {
    if (options && a === "--") options = false;
    else if (options && a.startsWith("-") && a !== "-") continue;
    else out.push(a);
  }
  return out;
}

/** Whether a short-flag cluster (`-rf`, `-Rv`) or long flag is present. */
function hasFlag(args: string[], short: string, long?: string): boolean {
  return args.some((a) =>
    (long !== undefined && a === long) ||
    (/^-[A-Za-z]+$/.test(a) && a.slice(1).includes(short))
  );
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

class Findings {
  readonly block: string[] = [];
  readonly warn: string[] = [];
}

type Rule = (cmd: Effective, f: Findings) => void;

function ruleRm(cmd: Effective, f: Findings): void {
  if (cmd.args.includes("--no-preserve-root")) {
    f.block.push("rm --no-preserve-root");
  }
  const recursive = hasFlag(cmd.args, "r", "--recursive") ||
    hasFlag(cmd.args, "R");
  if (!recursive) return;
  for (const target of operands(cmd.args)) {
    if (isCriticalPath(target)) {
      f.block.push(`recursive rm of ${target}`);
    }
  }
}

function ruleMkfs(cmd: Effective, f: Findings): void {
  f.block.push(`${cmd.name} formats a filesystem`);
}

function ruleDd(cmd: Effective, f: Findings): void {
  for (const a of cmd.args) {
    if (!a.startsWith("of=/dev/")) continue;
    if (!HARMLESS_DEVICE.test(a.slice(3))) {
      f.block.push(`dd writing to device ${a.slice(3)}`);
    }
  }
}

const DISKUTIL_DESTRUCTIVE =
  /^(erase\w*|secureerase|zerodisk|randomdisk|partitiondisk|reformat)$/;

function ruleDiskutil(cmd: Effective, f: Findings): void {
  const [verb = "", sub = ""] = operands(cmd.args).map((a) => a.toLowerCase());
  if (DISKUTIL_DESTRUCTIVE.test(verb)) f.block.push(`diskutil ${verb}`);
  if (verb === "apfs" && /^(delete\w*|erase\w*)$/.test(sub)) {
    f.block.push(`diskutil apfs ${sub}`);
  }
}

function rulePermissions(cmd: Effective, f: Findings): void {
  const recursive = hasFlag(cmd.args, "R", "--recursive");
  // First operand is the mode / owner, unless --reference= supplies it.
  const ops = operands(cmd.args);
  const targets = cmd.args.some((a) => a.startsWith("--reference="))
    ? ops
    : ops.slice(1);
  for (const t of targets) {
    if (normalizePath(t) === "/" || (recursive && isCriticalPath(t))) {
      f.block.push(`${recursive ? "recursive " : ""}${cmd.name} on ${t}`);
    }
  }
}

/** Git global options that take a separate argument. */
const GIT_OPTS_WITH_ARG = new Set([
  "-C",
  "-c",
  "--git-dir",
  "--work-tree",
  "--namespace",
  "--exec-path",
]);

function gitSubcommand(args: string[]): [string, string[]] {
  let i = 0;
  while (i < args.length && args[i].startsWith("-")) {
    i += GIT_OPTS_WITH_ARG.has(args[i]) ? 2 : 1;
  }
  return [args[i] ?? "", args.slice(i + 1)];
}

function ruleGit(cmd: Effective, f: Findings): void {
  const [sub, rest] = gitSubcommand(cmd.args);
  if (sub === "push") {
    const forced = hasFlag(rest, "f", "--force") ||
      operands(rest).some((a) => a.startsWith("+"));
    if (forced) f.warn.push("git push --force rewrites remote history");
  } else if (sub === "reset" && rest.includes("--hard")) {
    f.warn.push("git reset --hard discards uncommitted changes");
  } else if (sub === "clean" && hasFlag(rest, "f", "--force")) {
    f.warn.push("git clean -f deletes untracked files");
  }
}

function ruleFind(cmd: Effective, f: Findings): void {
  if (cmd.args.includes("-delete")) f.warn.push("find -delete");
  cmd.args.forEach((a, i) => {
    if (["-exec", "-execdir", "-ok", "-okdir"].includes(a)) {
      if (basename(cmd.args[i + 1] ?? "") === "rm") {
        f.warn.push("find -exec rm");
      }
    }
  });
}

function rulePower(cmd: Effective, f: Findings): void {
  f.warn.push(`${cmd.name} stops or restarts the machine`);
}

function ruleInit(cmd: Effective, f: Findings): void {
  if (["0", "6"].includes(cmd.args[0] ?? "")) {
    f.warn.push(`${cmd.name} ${cmd.args[0]} stops or restarts the machine`);
  }
}

function ruleSystemctl(cmd: Effective, f: Findings): void {
  const verb = operands(cmd.args)[0] ?? "";
  if (["poweroff", "reboot", "halt", "kexec", "suspend"].includes(verb)) {
    f.warn.push(`systemctl ${verb} stops or restarts the machine`);
  }
}

function ruleKill(cmd: Effective, f: Findings): void {
  const args = [...cmd.args];
  // Drop the signal spec: `-9`, `-KILL`, `-s KILL`, `-n 9`.
  if (args[0] === "-s" || args[0] === "-n") args.splice(0, 2);
  else if (args[0] !== "--" && /^-\w+$/.test(args[0] ?? "")) args.shift();
  if (args.includes("-1")) {
    f.warn.push("kill -1 signals every process you can reach");
  }
}

function ruleCrontab(cmd: Effective, f: Findings): void {
  if (hasFlag(cmd.args, "r")) {
    f.warn.push("crontab -r deletes the whole crontab");
  }
}

function ruleHistory(cmd: Effective, f: Findings): void {
  if (hasFlag(cmd.args, "c")) f.warn.push("history -c clears shell history");
}

function ruleTee(cmd: Effective, f: Findings): void {
  const append = hasFlag(cmd.args, "a", "--append");
  for (const t of operands(cmd.args)) {
    if (RAW_DISK.test(t)) f.block.push(`write to raw disk device ${t}`);
    else if (!append && SHELL_RC.test(t)) f.warn.push(`overwrites ${t}`);
  }
}

function ruleCopy(cmd: Effective, f: Findings): void {
  const ops = operands(cmd.args);
  const dest = ops.length > 1 ? ops[ops.length - 1] : undefined;
  if (dest === undefined) return;
  if (RAW_DISK.test(dest)) f.block.push(`write to raw disk device ${dest}`);
  else if (SHELL_RC.test(dest)) f.warn.push(`overwrites ${dest}`);
}

const RULES: Record<string, Rule> = {
  rm: ruleRm,
  mkfs: ruleMkfs,
  mke2fs: ruleMkfs,
  mkswap: ruleMkfs,
  wipefs: ruleMkfs,
  dd: ruleDd,
  diskutil: ruleDiskutil,
  chmod: rulePermissions,
  chown: rulePermissions,
  chgrp: rulePermissions,
  git: ruleGit,
  find: ruleFind,
  shutdown: rulePower,
  reboot: rulePower,
  halt: rulePower,
  poweroff: rulePower,
  init: ruleInit,
  telinit: ruleInit,
  systemctl: ruleSystemctl,
  kill: ruleKill,
  crontab: ruleCrontab,
  history: ruleHistory,
  tee: ruleTee,
  cp: ruleCopy,
  mv: ruleCopy,
};

function ruleFor(name: string): Rule | undefined {
  if (/^mkfs(\..+)?$/.test(name) || /^newfs(_\w+)?$/.test(name)) {
    return ruleMkfs;
  }
  return Object.hasOwn(RULES, name) ? RULES[name] : undefined;
}

function checkRedirects(redirects: Redirect[], f: Findings): void {
  for (const { op, target } of redirects) {
    if (!op.includes(">") || op === ">&") continue;
    if (RAW_DISK.test(target)) {
      f.block.push(`write to raw disk device ${target}`);
    } else if (!op.includes(">>") && SHELL_RC.test(target)) {
      f.warn.push(`overwrites ${target}`);
    }
  }
}

const MAX_DEPTH = 4;

/** Script text a shell or `eval` will run, if any. */
function nestedScript(cmd: Effective): string | null {
  if (cmd.name === "eval") return cmd.args.join(" ");
  if (!SHELLS.has(cmd.name)) return null;
  const c = cmd.args.findIndex((a) => /^-[a-z]*c[a-z]*$/.test(a));
  return c === -1 ? null : cmd.args[c + 1] ?? null;
}

function checkWrappers(cmd: Effective, f: Findings): void {
  if (cmd.wrappers.includes("sudo") || cmd.wrappers.includes("doas")) {
    f.warn.push("runs with elevated privileges (sudo)");
  }
  if (cmd.wrappers.includes("xargs") && cmd.name === "rm") {
    f.warn.push("xargs rm deletes files from piped input");
  }
}

/** `bash <(curl …)`, `sh -c "$(wget …)"`, `eval "$(curl …)"`. */
function checkDownloadedScript(cmd: Effective, f: Findings): void {
  const runsScripts = SHELLS.has(cmd.name) || cmd.name === "eval" ||
    cmd.name === "source";
  if (
    runsScripts &&
    /(\$|<)\((\s*sudo)?\s*(curl|wget)\b/.test(cmd.args.join(" "))
  ) {
    f.warn.push("runs a downloaded script (curl/wget into a shell)");
  }
}

function checkSimple(sc: SimpleCommand, f: Findings, depth: number): void {
  checkRedirects(sc.redirects, f);
  const cmd = effective(sc.argv);
  if (!cmd) return;
  checkWrappers(cmd, f);
  ruleFor(cmd.name)?.(cmd, f);
  checkDownloadedScript(cmd, f);
  const script = nestedScript(cmd);
  if (script !== null && depth < MAX_DEPTH) classifyInto(script, f, depth + 1);
}

function checkPipeline(pipeline: SimpleCommand[], f: Findings): void {
  let downloading = false;
  for (const sc of pipeline) {
    const cmd = effective(sc.argv);
    if (!cmd) continue;
    if (downloading && SHELLS.has(cmd.name)) {
      f.warn.push("pipes a download into a shell (curl | sh)");
      return;
    }
    if (cmd.name === "curl" || cmd.name === "wget") downloading = true;
  }
}

/** Raw-text checks that apply even inside quotes. */
function checkRaw(command: string, f: Findings): void {
  const compact = command.replace(/\s+/g, "");
  if (/([A-Za-z_:][\w:]*)\(\)\{\1\|\1&\};?\1/.test(compact)) {
    f.block.push("fork bomb");
  }
  const sql = command.match(
    /\b(DROP\s+(TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE)\b/i,
  );
  if (sql) f.warn.push(`SQL ${sql[1].replace(/\s+/g, " ").toUpperCase()}`);
}

function classifyInto(command: string, f: Findings, depth: number): void {
  const parsed = new Lexer(command).parse();
  for (const pipeline of parsed.pipelines) {
    checkPipeline(pipeline, f);
    for (const sc of pipeline) checkSimple(sc, f, depth);
  }
  if (depth >= MAX_DEPTH) return;
  for (const body of parsed.substitutions) classifyInto(body, f, depth + 1);
}

/**
 * Classify a shell command as `block`, `warn` or `ok`, with the reasons.
 *
 * Block: recursive `rm` of `/`, `~`, `$HOME`, `*` or a system root (and
 * those with `/*`), `rm --no-preserve-root`; `mkfs*` / `mke2fs` / `newfs*` /
 * `mkswap` / `wipefs`; `dd of=/dev/…` (other than `/dev/null` and friends);
 * redirects, `tee`, `cp` or `mv` onto a raw disk device (`/dev/sda`,
 * `/dev/disk2`, `/dev/nvme0n1`, …); fork bombs; `diskutil erase*` /
 * `secureErase` / `zeroDisk` / `randomDisk` / `partitionDisk`; `chmod` /
 * `chown` / `chgrp` on `/`, or recursively on any of the `rm` targets.
 *
 * Warn: `sudo` / `doas`; `curl` / `wget` piped or substituted into a shell;
 * `git push --force` (not `--force-with-lease`), `git reset --hard`,
 * `git clean -f`; `find -delete` / `find -exec rm`; `xargs rm`;
 * `shutdown` / `reboot` / `halt` / `poweroff` / `init 0|6` /
 * `systemctl reboot…`; `kill … -1`; SQL `DROP TABLE|DATABASE|SCHEMA` /
 * `TRUNCATE TABLE`; `crontab -r`; overwriting a shell rc file (`>`, `tee`,
 * `cp`, `mv`; appending with `>>` is ok); `history -c`.
 *
 * See the module docs for how quoting is treated: `echo "rm -rf /"` is `ok`,
 * `bash -c "rm -rf /"` is `block`.
 *
 * @example
 * ```ts
 * classifyCommand("sudo rm -rf / --no-preserve-root").verdict; // "block"
 * classifyCommand("git push --force origin main").verdict; // "warn"
 * classifyCommand("rm -rf ./build").verdict; // "ok"
 * ```
 */
export function classifyCommand(command: string): CommandClassification {
  const f = new Findings();
  checkRaw(command, f);
  classifyInto(command, f, 0);
  const block = [...new Set(f.block)];
  const warn = [...new Set(f.warn)];
  const verdict: CommandVerdict = block.length > 0
    ? "block"
    : warn.length > 0
    ? "warn"
    : "ok";
  return { verdict, reasons: [...block, ...warn] };
}
