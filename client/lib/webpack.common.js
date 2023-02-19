const path = require("path");
const webpack = require("webpack");
module.exports = {
    entry: "./src/index.ts",
    resolve: {
        extensions: [".ts", ".js"],
        fallback: {
            // "buffer": false
        }
    },
    module: {
        rules: [
            {
                test: [/\.ts$/, /\.tsx$/],
                use: [
                    {
                        loader: "ts-loader",
                        options: {
                            configFile: "tsconfig.json",
                        },
                    },
                ],
            },
        ],
    },
    output: {
        filename: "index.js",
        path: path.resolve(__dirname, "dist"),
        libraryTarget: "umd",
        globalObject: "typeof self !== 'undefined' ? self : this",
    },
    externals: {
        react: "react",
        "react-dom": "reactDOM",
    }
};
