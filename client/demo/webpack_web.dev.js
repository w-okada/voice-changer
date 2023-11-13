const path = require("path");
const { merge } = require("webpack-merge");
const common = require("./webpack_web.common.js");
const express = require("express");
module.exports = merge(common, {
    mode: "development",
    devServer: {
        setupMiddlewares: (middlewares, devServer) => {
            if (!devServer) {
                throw new Error("webpack-dev-server is not defined");
            }

            // ミドルウェアを追加して静的ファイルへのアクセスログを出力
            devServer.app.use(
                "/",
                express.static(path.join(__dirname, "dist_web"), {
                    setHeaders: (res, filepath) => {
                        console.log(`Serving static file: ${filepath}`);
                    },
                }),
            );

            // 既存のミドルウェアをそのまま利用
            return middlewares;
        },
        client: {
            overlay: {
                errors: false,
                warnings: false,
            },
            logging: "log",
        },
        host: "0.0.0.0",
        https: true,
    },
});
