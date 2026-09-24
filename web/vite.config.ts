import { defineConfig } from "vite";
import fs from "fs";
import path from "path";

export default defineConfig({
  server: {
    port: 5173,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  plugins: [
    {
      name: "serve-ort-assets",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const rawUrl = req.url?.split("?")[0] || "";
          if (rawUrl.startsWith("/ort-wasm") || rawUrl.startsWith("/ort.")) {
            const fileName = rawUrl.replace(/^\//, "");
            const filePath = path.resolve(__dirname, "node_modules/onnxruntime-web/dist", fileName);
            if (fs.existsSync(filePath)) {
              if (fileName.endsWith(".wasm")) {
                res.setHeader("Content-Type", "application/wasm");
              } else if (fileName.endsWith(".mjs") || fileName.endsWith(".js")) {
                res.setHeader("Content-Type", "application/javascript");
              }
              res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
              res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
              fs.createReadStream(filePath).pipe(res);
              return;
            }
          }
          next();
        });
      },
    },
  ],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});

