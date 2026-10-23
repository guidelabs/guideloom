import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { context } from "esbuild";
import { compile } from "svelte/compiler";

const directory = path.dirname(fileURLToPath(import.meta.url));
const destination = path.resolve(directory, "../assets/generated/storage-formats.js");

async function writeIfChanged(filename, contents) {
  try {
    if (await readFile(filename, "utf8") === contents) return;
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, contents);
}

async function writeAssets(result) {
  if (result.errors.length) return;
  const packages = [...new Set(Object.keys(result.metafile.inputs)
    .map(filename => filename.match(/^node_modules\/((?:@[^/]+\/)?[^/]+)/)?.[1])
    .filter(Boolean))].sort();
  const notices = await Promise.all(packages.map(async name => {
    const packageDirectory = path.join(directory, "node_modules", name);
    const license = (await readdir(packageDirectory)).find(filename => /^license(?:\.|$)/i.test(filename));
    if (!license) throw new Error(`Missing license for bundled package ${name}`);
    return `${name}\n${await readFile(path.join(packageDirectory, license), "utf8")}`;
  }));
  await writeIfChanged(destination.replace(/\.js$/, ".LICENSE.txt"), notices.join("\n\n"));
  await writeIfChanged(destination, result.outputFiles[0].text);
  console.log("Built storage format visualization.");
}

export async function buildComponents({ watch = false } = {}) {
  const builder = await context({
    absWorkingDir: directory,
    entryPoints: ["index.js"],
    bundle: true,
    minify: true,
    format: "iife",
    platform: "browser",
    target: "es2022",
    conditions: ["browser"],
    define: { "process.env.NODE_ENV": '"production"' },
    banner: { js: "// Generated from docs/components/. Do not edit." },
    legalComments: "inline",
    metafile: true,
    write: false,
    plugins: [{
      name: "svelte",
      setup(build) {
        build.onLoad({ filter: /\.svelte$/ }, async ({ path: filename }) => {
          const { js, warnings } = compile(await readFile(filename, "utf8"), {
            filename: path.relative(directory, filename),
            generate: "client",
            css: "injected",
            discloseVersion: false,
          });
          if (warnings.length) {
            throw new Error(warnings.map(warning => `${warning.code}: ${warning.message}`).join("\n"));
          }
          return { contents: js.code, resolveDir: path.dirname(filename) };
        });
        build.onEnd(writeAssets);
      },
    }],
  });
  try {
    await builder.rebuild();
    if (watch) await builder.watch();
    else await builder.dispose();
  } catch (error) {
    await builder.dispose();
    throw error;
  }
  return builder;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await buildComponents();
}
