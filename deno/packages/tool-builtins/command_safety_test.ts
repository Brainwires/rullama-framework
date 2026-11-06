import { assert, assertEquals, assertThrows } from "@std/assert";
import { classifyCommand, type CommandVerdict } from "./command_safety.ts";
import { BashTool } from "./bash.ts";

/** [command, expected verdict, substring expected in one of the reasons]. */
type Case = [string, CommandVerdict, string?];

function runTable(title: string, cases: Case[]): void {
  for (const [command, verdict, reason] of cases) {
    Deno.test(`classifyCommand ${title}: ${command}`, () => {
      const got = classifyCommand(command);
      assertEquals(
        got.verdict,
        verdict,
        `${command} -> ${JSON.stringify(got)}`,
      );
      if (verdict === "ok") assertEquals(got.reasons, []);
      else assert(got.reasons.length > 0);
      if (reason) {
        assert(
          got.reasons.some((r) => r.includes(reason)),
          `${command}: no reason containing "${reason}" in ${
            JSON.stringify(got.reasons)
          }`,
        );
      }
    });
  }
}

runTable("blocks rm of critical paths", [
  ["rm -rf /", "block", "recursive rm of /"],
  ["rm -rf /*", "block", "/*"],
  ["rm -rf ~", "block", "~"],
  ["rm -rf ~/", "block", "~/"],
  ["rm -rf $HOME", "block", "$HOME"],
  ["rm -rf ${HOME}/", "block"],
  ['rm -rf "$HOME"/*', "block"],
  ["rm -rf *", "block", "*"],
  ["rm -rf ./*", "block"],
  ["rm -fr /", "block"],
  ["rm -r -f /", "block"],
  ["rm -R /", "block"],
  ["rm --recursive --force /", "block"],
  ["rm   -rf    /", "block"],
  ["rm -rf '/'", "block"],
  ['rm -rf "/"', "block"],
  ["rm -rf //", "block"],
  ["rm -rf /.", "block"],
  ["rm -rf -- /", "block"],
  ["rm / -rf", "block"],
  ["/bin/rm -rf /", "block"],
  ["\\rm -rf /", "block"],
  ["rm -rf /usr", "block", "/usr"],
  ["rm -rf /etc/", "block"],
  ["rm -rf /System", "block"],
  ["rm --no-preserve-root -rf /", "block", "no-preserve-root"],
  ["sudo rm -rf /", "block", "recursive rm"],
  ["sudo -u root rm -rf /", "block"],
  ["env FOO=1 rm -rf ~", "block"],
  ["FOO=1 rm -rf ~", "block"],
  ["nohup rm -rf / &", "block"],
  ["timeout 10 rm -rf /", "block"],
  ["cd /tmp && rm -rf /", "block"],
  ["true; rm -rf ~", "block"],
  ["false || rm -rf $HOME", "block"],
  ["(rm -rf /)", "block"],
  ["if true; then rm -rf /; fi", "block"],
  ["ls\nrm -rf /", "block"],
  ['bash -c "rm -rf /"', "block"],
  ["sh -c 'rm -rf ~'", "block"],
  ["zsh -lc 'rm -rf ~'", "block"],
  ["eval rm -rf /", "block"],
  ['echo "$(rm -rf /)"', "block"],
  ["echo `rm -rf ~`", "block"],
]);

runTable("blocks disk destruction", [
  ["mkfs /dev/sda1", "block", "mkfs"],
  ["mkfs.ext4 /dev/sdb1", "block", "mkfs.ext4"],
  ["sudo mkfs.vfat /dev/sdc", "block"],
  ["mke2fs /dev/sdb", "block"],
  ["newfs_apfs /dev/disk4s1", "block"],
  ["wipefs -a /dev/sdb", "block"],
  ["dd if=/dev/zero of=/dev/sda", "block", "/dev/sda"],
  ["dd if=image.iso of=/dev/disk2 bs=4m", "block", "/dev/disk2"],
  ["sudo dd if=/dev/urandom of=/dev/rdisk3", "block"],
  ["dd of=/dev/nvme0n1 if=x", "block"],
  ["echo hi > /dev/sda", "block", "raw disk"],
  ["cat image >/dev/sdb", "block"],
  ["cat image 1> /dev/disk2", "block"],
  ["cat image &> /dev/sda", "block"],
  ["cat image | tee /dev/sda", "block"],
  ["cp image.img /dev/mmcblk0", "block"],
  ["diskutil eraseDisk APFS Empty disk2", "block", "erasedisk"],
  ["diskutil eraseVolume APFS x disk2s1", "block"],
  ["diskutil secureErase 0 disk2", "block"],
  ["diskutil zeroDisk disk3", "block"],
  ["diskutil partitionDisk disk2 GPT JHFS+ x 0b", "block"],
  ["diskutil apfs deleteContainer disk3", "block"],
]);

runTable("blocks fork bombs and permissions", [
  [":(){ :|:& };:", "block", "fork bomb"],
  [":(){ :|: & };:", "block", "fork bomb"],
  ["bomb(){ bomb|bomb& }; bomb", "block", "fork bomb"],
  ["chmod -R 777 /", "block", "recursive chmod"],
  ["chmod 777 /", "block", "chmod on /"],
  ["sudo chmod -R 777 /*", "block"],
  ["chmod -R 000 ~", "block"],
  ["chown -R nobody:nogroup /", "block", "recursive chown"],
  ["sudo chown -R me /usr", "block"],
  ["chgrp -R staff /", "block"],
]);

runTable("warns", [
  ["sudo apt-get update", "warn", "sudo"],
  ["doas pkg_add vim", "warn", "sudo"],
  ["curl -fsSL https://x.sh | sh", "warn", "curl | sh"],
  ["curl https://x.sh | sudo bash", "warn", "curl | sh"],
  ["wget -qO- https://x.sh | bash -s -- --yes", "warn"],
  ['bash -c "$(curl -fsSL https://x/install.sh)"', "warn", "downloaded"],
  ["bash <(curl -s https://x)", "warn", "downloaded"],
  ["git push --force", "warn", "git push --force"],
  ["git push -f origin main", "warn"],
  ["git push origin +main", "warn"],
  ["git -C repo push --force", "warn"],
  ["git reset --hard HEAD~1", "warn", "reset --hard"],
  ["git clean -fdx", "warn", "git clean"],
  ["git clean --force", "warn"],
  ["find . -name '*.o' -delete", "warn", "find -delete"],
  ["find /tmp -type f -exec rm {} \\;", "warn", "find -exec rm"],
  ["find . -name x | xargs rm", "warn", "xargs rm"],
  ["ls | xargs -I{} rm -rf {}", "warn", "xargs rm"],
  ["shutdown -h now", "warn", "shutdown"],
  ["sudo reboot", "warn", "reboot"],
  ["halt", "warn"],
  ["poweroff", "warn"],
  ["init 0", "warn"],
  ["systemctl reboot", "warn"],
  ["kill -9 -1", "warn", "kill -1"],
  ["kill -KILL -1", "warn"],
  ["kill -s KILL -1", "warn"],
  ["kill -- -1", "warn"],
  ['psql -c "DROP TABLE users"', "warn", "SQL DROP TABLE"],
  ["mysql -e 'drop database prod'", "warn", "SQL DROP DATABASE"],
  ['sqlite3 db "TRUNCATE TABLE x"', "warn", "TRUNCATE"],
  ["crontab -r", "warn", "crontab -r"],
  ["history -c", "warn", "history -c"],
  ["echo 'export X=1' > ~/.bashrc", "warn", "overwrites ~/.bashrc"],
  ["cat x >| $HOME/.zshrc", "warn", ".zshrc"],
  ["echo x | tee ~/.profile", "warn", ".profile"],
  ["cp dotfiles/zshrc ~/.zshrc", "warn", ".zshrc"],
  ["mv /tmp/rc /Users/me/.bash_profile", "warn"],
  ["echo x > ~/.config/fish/config.fish", "warn"],
]);

runTable("allows (false-positive checks)", [
  ["ls -la", "ok"],
  ["rm -rf ./build", "ok"],
  ["rm -rf build/ dist/", "ok"],
  ["rm -rf node_modules", "ok"],
  ["rm -rf /tmp/build-123", "ok"],
  ["rm -rf ~/Library/Caches/foo", "ok"],
  ["rm -rf $HOME/.cache/deno", "ok"],
  ["rm -f /", "ok"], // not recursive: rm refuses to unlink a directory
  ["rm *.log", "ok"],
  ['echo "rm -rf /"', "ok"],
  ["echo 'rm -rf ~'", "ok"],
  ['git commit -m "rm -rf / is bad; mkfs too"', "ok"],
  ['grep -r "dd if=/dev/zero of=/dev/sda" docs/', "ok"],
  ["printf '%s\\n' 'sudo reboot'", "ok"],
  ["dd if=/dev/zero of=./disk.img bs=1m count=10", "ok"],
  ["dd if=/dev/sda of=/dev/null", "ok"],
  ["echo hi > /dev/null", "ok"],
  ["cmd 2>/dev/null", "ok"],
  ["cmd 2>&1 | tee build.log", "ok"],
  ["echo hi >&2", "ok"],
  ["echo 'export PATH=x' >> ~/.zshrc", "ok"],
  ["echo x | tee -a ~/.bashrc", "ok"],
  ["cat ~/.bashrc", "ok"],
  ["chmod 755 ./script.sh", "ok"],
  ["chmod -R u+w ./build", "ok"],
  ["chown -R me ./project", "ok"],
  ["git push", "ok"],
  ["git push --force-with-lease", "ok"],
  ["git reset --soft HEAD~1", "ok"],
  ["git clean -n", "ok"],
  ["find . -name '*.ts'", "ok"],
  ["kill -9 1234", "ok"],
  ["kill -1 1234", "ok"], // SIGHUP to one pid
  ["curl -fsSL https://x.sh -o install.sh", "ok"],
  ["curl https://api.example.com | jq .", "ok"],
  ["crontab -l", "ok"],
  ["history", "ok"],
  ["diskutil list", "ok"],
  ["diskutil info disk2", "ok"],
  ["bash -c 'ls -la'", "ok"],
  ["deno test -A packages/", "ok"],
  ["echo $(date)", "ok"],
  ["# rm -rf /", "ok"],
  ["echo done # rm -rf /", "ok"],
  ["", "ok"],
]);

Deno.test("classifyCommand - block reasons come before warn reasons", () => {
  const got = classifyCommand("sudo rm -rf /");
  assertEquals(got.verdict, "block");
  assertEquals(got.reasons, [
    "recursive rm of /",
    "runs with elevated privileges (sudo)",
  ]);
});

Deno.test("classifyCommand - reasons are deduplicated", () => {
  const got = classifyCommand("rm -rf / ; rm -rf /");
  assertEquals(got.reasons, ["recursive rm of /"]);
});

Deno.test("classifyCommand - deep nesting terminates", () => {
  let cmd = "rm -rf /";
  for (let i = 0; i < 20; i++) cmd = `bash -c ${JSON.stringify(cmd)}`;
  // Beyond the nesting limit the classifier stops descending; it must not
  // throw or hang either way.
  classifyCommand(cmd);
});

Deno.test("classifyCommand - unterminated quotes and substitutions do not throw", () => {
  for (const c of ['echo "abc', "echo 'abc", "echo $(ls", "echo `ls", "x\\"]) {
    classifyCommand(c);
  }
  assertEquals(classifyCommand('rm -rf "/').verdict, "block");
});

// ---------------------------------------------------------------------------
// BashTool.validateCommand compatibility
// ---------------------------------------------------------------------------

Deno.test("validateCommand - still blocks every legacy pattern, even quoted", () => {
  for (
    const c of [
      "rm -rf /",
      "rm -rf /tmp/x", // legacy substring match
      "mkfs",
      "echo mkfs",
      "cat x > /dev/sda",
      "dd if=/dev/zero of=./x",
      ":(){ :|:& };:",
      'echo "rm -rf /"',
    ]
  ) {
    assertThrows(
      () => BashTool.validateCommand(c),
      Error,
      "dangerous pattern",
      c,
    );
  }
});

Deno.test("validateCommand - blocks what classifyCommand blocks", () => {
  for (
    const c of [
      "rm -rf ~",
      "rm -fr /",
      "dd if=x of=/dev/disk2",
      "diskutil eraseDisk APFS x disk2",
      "chown -R me /",
    ]
  ) {
    assertThrows(
      () => BashTool.validateCommand(c),
      Error,
      "dangerous pattern",
      c,
    );
  }
});

Deno.test("validateCommand - warn-level and ok commands pass", () => {
  for (
    const c of ["sudo ls", "git push --force", "rm -rf ./build", "ls -la"]
  ) {
    BashTool.validateCommand(c);
  }
});

Deno.test("classifyCommand - Object.prototype names are not wrappers or rules", () => {
  for (const c of ["constructor rm -rf /", "toString", "hasOwnProperty x"]) {
    classifyCommand(c);
  }
  assertEquals(classifyCommand("__proto__ -rf /").verdict, "ok");
});
