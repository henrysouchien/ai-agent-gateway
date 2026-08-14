import fs from "node:fs";
import ts from "typescript";

const input = process.argv[2];
const source = fs.readFileSync(input, "utf8");
const ast = ts.createSourceFile(input, source, ts.ScriptTarget.ES2020, true, ts.ScriptKind.TSX);
const allowedImports = new Set(["react", "recharts", "@hank/canvas-kit"]);
const forbidden = new Set([
  "fetch", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts", "eval", "Function",
  "localStorage", "sessionStorage", "indexedDB", "setTimeout", "setInterval", "postMessage",
]);
const diagnostics = [];
function add(node, code, message, repair_hint) {
  const pos = ast.getLineAndCharacterOfPosition(node.getStart(ast));
  diagnostics.push({ line: pos.line + 1, column: pos.character + 1, code, message, repair_hint });
}
let defaults = 0;
let defaultIsComponent = false;
const componentNames = new Set();
for (const statement of ast.statements) {
  if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement)) && statement.name) componentNames.add(statement.name.text);
  if (ts.isVariableStatement(statement)) for (const declaration of statement.declarationList.declarations) {
    if (ts.isIdentifier(declaration.name) && declaration.initializer && (ts.isArrowFunction(declaration.initializer) || ts.isFunctionExpression(declaration.initializer) || ts.isClassExpression(declaration.initializer))) componentNames.add(declaration.name.text);
  }
}
for (const statement of ast.statements) {
  const modifiers = ts.canHaveModifiers(statement) ? ts.getModifiers(statement) ?? [] : [];
  if (modifiers.some(item => item.kind === ts.SyntaxKind.DefaultKeyword)) {
    defaults += 1;
    if (ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement)) defaultIsComponent = true;
  }
  if (ts.isExportAssignment(statement) && !statement.isExportEquals) {
    defaults += 1;
    if (ts.isIdentifier(statement.expression) && componentNames.has(statement.expression.text)) defaultIsComponent = true;
  }
  if (ts.isImportDeclaration(statement)) {
    const specifier = ts.isStringLiteral(statement.moduleSpecifier) ? statement.moduleSpecifier.text : "";
    if (!allowedImports.has(specifier)) add(statement, "import_allowlist", `Import '${specifier}' is not allowed.`, "Import only react, recharts, or @hank/canvas-kit.");
    if (statement.importClause?.namedBindings && ts.isNamespaceImport(statement.importClause.namedBindings)) add(statement, "import_structure", "Namespace imports are not supported by the frozen externals ABI.", "Use default or named imports from the three allowed modules.");
    continue;
  }
  if (ts.isVariableStatement(statement)) {
    if (!(statement.declarationList.flags & ts.NodeFlags.Const)) add(statement, "module_scope_side_effect", "Only const declarations are allowed at module scope.", "Replace mutable module state with component-local computation.");
    for (const declaration of statement.declarationList.declarations) if (declaration.initializer) {
      function rejectCall(node) {
        if (ts.isCallExpression(node) || ts.isNewExpression(node) || ts.isAwaitExpression(node)) add(node, "module_scope_side_effect", "Calls and side effects are forbidden in module-scope initializers.", "Keep top-level constants as literal analytical data.");
        ts.forEachChild(node, rejectCall);
      }
      rejectCall(declaration.initializer);
    }
    continue;
  }
  if (ts.isTypeAliasDeclaration(statement) || ts.isInterfaceDeclaration(statement) || ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement) || ts.isExportDeclaration(statement) || statement.kind === ts.SyntaxKind.EmptyStatement) continue;
  if (ts.isExportAssignment(statement)) continue;
  add(statement, "module_scope_side_effect", "Executable statement is not allowed at module scope.", "Keep module scope to imports, types, constants, and the component declaration.");
}
if (defaults !== 1) add(ast, "default_export_count", "Canvas source must contain exactly one default export.", "Export one React component as default.");
else if (!defaultIsComponent) add(ast, "default_export_component", "The default export must structurally resolve to a component function or class.", "Default-export a function, class, or component-valued const.");
function visit(node) {
  if (ts.isIdentifier(node) && forbidden.has(node.text)) add(node, "forbidden_identifier", `Forbidden Canvas identifier: ${node.text}.`, "Remove browser, timer, storage, dynamic-code, and messaging APIs.");
  if (ts.isPropertyAccessExpression(node) && node.expression.getText(ast) === "document" && node.name.text === "cookie") add(node, "forbidden_identifier", "Forbidden Canvas identifier: document.cookie.", "Remove browser storage access.");
  if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) add(node, "forbidden_identifier", "Dynamic import() is forbidden.", "Use only static imports from the Canvas runtime contract.");
  ts.forEachChild(node, visit);
}
visit(ast);
process.stdout.write(JSON.stringify(diagnostics.slice(0, 8)));
process.exitCode = diagnostics.length ? 2 : 0;
