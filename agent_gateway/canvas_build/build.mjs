import fs from "node:fs";
import path from "node:path";
import ts from "typescript";
import * as esbuild from "esbuild";

const [inputPath, outputPath, contractDir] = process.argv.slice(2);
if (!inputPath || !outputPath || !contractDir) throw new Error("input, output, and contract are required");
const source = fs.readFileSync(inputPath, "utf8");
const manifest = JSON.parse(fs.readFileSync(path.join(contractDir, "canvas_kit_manifest.v1.json"), "utf8"));
const ast = ts.createSourceFile(inputPath, source, ts.ScriptTarget.ES2020, true, ts.ScriptKind.TSX);
const requested = new Map();
for (const statement of ast.statements) {
  if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
  const specifier = statement.moduleSpecifier.text;
  const names = requested.get(specifier) ?? new Set();
  const clause = statement.importClause;
  if (clause?.name) names.add("default");
  if (clause?.namedBindings && ts.isNamedImports(clause.namedBindings)) {
    for (const element of clause.namedBindings.elements) names.add(element.propertyName?.text ?? element.name.text);
  }
  requested.set(specifier, names);
}
const externalsPlugin = {
  name: "hank-canvas-runtime-externals",
  setup(build) {
    build.onResolve({ filter: /.*/ }, args => manifest.externals[args.path] ? { path: args.path, namespace: "hank-runtime" } : null);
    build.onLoad({ filter: /.*/, namespace: "hank-runtime" }, args => {
      const globalPath = manifest.externals[args.path];
      const names = [...(requested.get(args.path) ?? [])];
      const exports = names.filter(name => name !== "default").map(name => `export const ${name}=runtime.${name};`).join("\n");
      const defaultExport = names.includes("default") ? "export default runtime;" : "";
      return { contents: `const runtime=${globalPath};\n${defaultExport}\n${exports}`, loader: "js" };
    });
  },
};
const compiler = manifest.bundle_format.compiler;
await esbuild.build({
  entryPoints: [inputPath],
  outfile: outputPath,
  bundle: true,
  platform: "browser",
  format: compiler.format,
  globalName: "__HankCanvasArtifact",
  jsx: compiler.jsx,
  target: compiler.target,
  charset: compiler.charset,
  legalComments: compiler.legalComments,
  minify: compiler.minify,
  sourcemap: false,
  plugins: [externalsPlugin],
  footer: { js: "HankCanvasRuntime.register(__HankCanvasArtifact.default);" },
  logLevel: "silent",
});
