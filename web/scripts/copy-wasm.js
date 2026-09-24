// Copy ONNX Runtime Web WASM binaries and companion loader modules from node_modules into public/ directory
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.resolve(__dirname, "../node_modules/onnxruntime-web/dist");
const dstDir = path.resolve(__dirname, "../public");

if (fs.existsSync(srcDir)) {
  fs.mkdirSync(dstDir, { recursive: true });
  let count = 0;
  for (const file of fs.readdirSync(srcDir)) {
    // Copy all WebAssembly engine artifacts: .wasm binaries and .mjs loader modules
    if (file.startsWith("ort-wasm") && (file.endsWith(".wasm") || file.endsWith(".mjs"))) {
      fs.copyFileSync(path.join(srcDir, file), path.join(dstDir, file));
      count++;
    }
  }
  console.log(`[StepView] Copied ${count} WASM/MJS asset(s) to public/`);
} else {
  console.warn(`[StepView] WASM source directory not found: ${srcDir}`);
}

