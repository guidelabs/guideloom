import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { buildComponents } from "./build.mjs";

const [command, ...args] = process.argv.slice(2);
if (command !== "build" && command !== "serve") {
  console.error("Usage: node docs.mjs <build|serve> [zensical options]");
  process.exit(1);
}

// Generate assets before Zensical discovers the documentation files.
const builder = await buildComponents({ watch: command === "serve" });
const child = spawn("python", ["-m", "zensical", command, ...args], {
  cwd: fileURLToPath(new URL("../../", import.meta.url)),
  stdio: "inherit",
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => child.kill(signal));
}
child.once("error", async error => {
  console.error(`Unable to start Zensical: ${error.message}. Run through uv or activate your Python environment.`);
  await builder.dispose();
  process.exitCode = 1;
});
child.once("exit", async (code, signal) => {
  await builder.dispose();
  process.exitCode = code ?? (signal === "SIGINT" ? 130 : 1);
});
