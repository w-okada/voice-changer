const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");
const webpack = require("webpack");
module.exports = {
    mode: "production",
    entry: "./src/000_index.tsx",
    resolve: {
        extensions: [".ts", ".tsx", ".js"],
        fallback: {
            buffer: require.resolve("buffer/"),
        },
    },
    module: {
        rules: [
            {
                test: [/\.ts$/, /\.tsx$/],
                use: [
                    {
                        loader: "babel-loader",
                        options: {
                            presets: ["@babel/preset-env", "@babel/preset-react", "@babel/preset-typescript"],
                            plugins: ["@babel/plugin-transform-runtime"],
                        },
                    },
                ],
            },
            {
                test: /\.html$/,
                loader: "html-loader",
            },
            {
                test: /\.css$/,
                use: ["style-loader", { loader: "css-loader", options: { importLoaders: 1 } }, "postcss-loader"],
            },
            { test: /\.json$/, type: "asset/inline" },
        ],
    },
    output: {
        filename: "index.js",
        path: path.resolve(__dirname, "dist"),
    },
    plugins: [
        new webpack.ProvidePlugin({
            Buffer: ["buffer", "Buffer"],
        }),
        new HtmlWebpackPlugin({
            template: path.resolve(__dirname, "public/index.html"),
            filename: "./index.html",
        }),
        new CopyPlugin({
            patterns: [{ from: "public/assets", to: "assets" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/favicon.ico", to: "favicon.ico" }],
        }),

        new CopyPlugin({
            patterns: [{ from: "./node_modules/@dannadori/voice-changer-js/dist/ort-wasm-simd.wasm", to: "ort-wasm-simd.wasm" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "./node_modules/@dannadori/voice-changer-js/dist/process.js", to: "process.js" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/emb_pit_24000.bin", to: "models/emb_pit_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvc2v_24000.bin", to: "models/rvc2v_24000.bin" }],
        }),
    ],
};
