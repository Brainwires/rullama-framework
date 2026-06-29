/**
 * @module config
 *
 * MCP server configuration management.
 * Equivalent to Rust's `rullama-mcp/src/config.rs`.
 */

/**
 * Configuration for a single MCP server.
 * Equivalent to Rust `McpServerConfig`.
 */
export interface McpServerConfig {
  /** Unique name for this server. */
  name: string;
  /** Command to launch the server. */
  command: string;
  /** Arguments to pass to the command. */
  args: string[];
  /** Optional environment variables. */
  env?: Record<string, string>;
}

/** Internal structure of the config file on disk. */
interface McpConfigFile {
  servers: McpServerConfig[];
}

/**
 * Manages MCP server configurations on disk.
 * Equivalent to Rust `McpConfigManager`.
 *
 * Stores configurations in `~/.rullama/mcp-config.json`.
 */
export class McpConfigManager {
  #configPath: string;
  #servers: McpServerConfig[];

  private constructor(configPath: string, servers: McpServerConfig[]) {
    this.#configPath = configPath;
    this.#servers = servers;
  }

  /**
   * Create a new config manager with empty servers list.
   * Equivalent to Rust `McpConfigManager::new`.
   */
  static create(): McpConfigManager {
    const configPath = McpConfigManager.getConfigPath();
    return new McpConfigManager(configPath, []);
  }

  /**
   * Load config from file, create default if it doesn't exist.
   * Equivalent to Rust `McpConfigManager::load`.
   */
  static async load(): Promise<McpConfigManager> {
    const configPath = McpConfigManager.getConfigPath();

    // Ensure config directory exists
    const dir = configPath.substring(0, configPath.lastIndexOf("/"));
    try {
      await Deno.mkdir(dir, { recursive: true });
    } catch (e) {
      if (!(e instanceof Deno.errors.AlreadyExists)) {
        throw new Error(`Failed to create config directory: ${e}`);
      }
    }

    // Load config file if it exists
    let servers: McpServerConfig[];
    try {
      const contents = await Deno.readTextFile(configPath);
      const config: McpConfigFile = JSON.parse(contents);
      servers = config.servers;
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) {
        // Create default config file
        const defaultConfig: McpConfigFile = { servers: [] };
        await Deno.writeTextFile(
          configPath,
          JSON.stringify(defaultConfig, null, 2),
        );
        servers = [];
      } else {
        throw new Error(`Failed to read MCP config file: ${e}`);
      }
    }

    return new McpConfigManager(configPath, servers);
  }

  /**
   * Save config to file.
   * Equivalent to Rust `McpConfigManager::save`.
   */
  async save(): Promise<void> {
    const config: McpConfigFile = { servers: this.#servers };
    const json = JSON.stringify(config, null, 2);
    await Deno.writeTextFile(this.#configPath, json);
  }

  /**
   * Add a new server configuration.
   * Equivalent to Rust `McpConfigManager::add_server`.
   */
  async addServer(config: McpServerConfig): Promise<void> {
    if (this.#servers.some((s) => s.name === config.name)) {
      throw new Error(`Server with name '${config.name}' already exists`);
    }
    this.#servers.push(config);
    await this.save();
  }

  /**
   * Remove a server configuration.
   * Equivalent to Rust `McpConfigManager::remove_server`.
   */
  async removeServer(name: string): Promise<void> {
    const initialLen = this.#servers.length;
    this.#servers = this.#servers.filter((s) => s.name !== name);
    if (this.#servers.length === initialLen) {
      throw new Error(`Server '${name}' not found`);
    }
    await this.save();
  }

  /**
   * Get all server configurations.
   * Equivalent to Rust `McpConfigManager::get_servers`.
   */
  getServers(): readonly McpServerConfig[] {
    return this.#servers;
  }

  /**
   * Get a specific server configuration.
   * Equivalent to Rust `McpConfigManager::get_server`.
   */
  getServer(name: string): McpServerConfig | undefined {
    return this.#servers.find((s) => s.name === name);
  }

  /**
   * Get the config file path.
   * Equivalent to Rust `McpConfigManager::get_config_path`.
   */
  private static getConfigPath(): string {
    const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE");
    if (!home) {
      throw new Error("Failed to get home directory");
    }
    return `${home}/.rullama/mcp-config.json`;
  }
}
