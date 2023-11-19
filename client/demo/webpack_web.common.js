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
            { test: /\.svg$/, type: "asset/resource" },
        ],
    },
    output: {
        filename: "index.js",
        path: path.resolve(__dirname, "dist_web"),
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

        // ダミーファイルコピー
        // new CopyPlugin({ //コピーの順番で上のassetのコピーで上書きされることがあるようだ。⇒npmスクリプトで対処。
        //     patterns: [{ from: "public/assets/gui_settings/edition_web.txt", to: "assets/gui_settings/edition.txt" }],
        // }),
        // new CopyPlugin({ // 拡張子なしのファイルコピーはできないようだ。⇒npmスクリプトで対処。
        //     patterns: [{ from: "public/info_web.txt", to: "info" }],
        // }),

        // VC用ファイルコピー
        new CopyPlugin({
            patterns: [{ from: "./node_modules/@dannadori/voice-changer-js/dist/ort-wasm-simd.wasm", to: "ort-wasm-simd.wasm" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "./node_modules/@dannadori/voice-changer-js/dist/tfjs-backend-wasm-simd.wasm", to: "tfjs-backend-wasm-simd.wasm" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "./node_modules/@dannadori/voice-changer-js/dist/process.js", to: "process.js" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv2_emb_pit_24000.bin", to: "models/rvcv2_emb_pit_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv2_amitaro_v2_32k_f0_24000.bin", to: "models/rvcv2_amitaro_v2_32k_f0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv2_amitaro_v2_32k_nof0_24000.bin", to: "models/rvcv2_amitaro_v2_32k_nof0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv2_amitaro_v2_40k_f0_24000.bin", to: "models/rvcv2_amitaro_v2_40k_f0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv2_amitaro_v2_40k_nof0_24000.bin", to: "models/rvcv2_amitaro_v2_40k_nof0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv1_emb_pit_24000.bin", to: "models/rvcv1_emb_pit_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv1_amitaro_v1_32k_f0_24000.bin", to: "models/rvcv1_amitaro_v1_32k_f0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv1_amitaro_v1_32k_nof0_24000.bin", to: "models/rvcv1_amitaro_v1_32k_nof0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv1_amitaro_v1_40k_f0_24000.bin", to: "models/rvcv1_amitaro_v1_40k_f0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/rvcv1_amitaro_v1_40k_nof0_24000.bin", to: "models/rvcv1_amitaro_v1_40k_nof0_24000.bin" }],
        }),
        new CopyPlugin({
            patterns: [{ from: "public/models/amitaro.png", to: "models/amitaro.png" }],
        }),
    ],
};
