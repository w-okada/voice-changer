/* eslint @typescript-eslint/no-var-requires: "off" */
const path = require("path");

module.exports = {
    // mode: "development",
    mode: "production",
    entry: path.resolve(__dirname, "worklet/src/voice-changer-worklet-processor.ts"),
    output: {
        path: path.resolve(__dirname, "worklet/dist"),
        filename: "index.js",
    },
    resolve: {
        modules: [path.resolve(__dirname, "node_modules")],
        extensions: [".ts", ".js"],
    },
    module: {
        rules: [
            {
                test: [/\.ts$/, /\.tsx$/],
                use: [
                    {
                        loader: "ts-loader",
                        options: {
                            configFile: "tsconfig.worklet.json",
                        },
                    },
                ],
            },
        ],
    }
};
