const path = require("path");
const { merge } = require('webpack-merge');
const common = require('./webpack.common.js')

module.exports = merge(common, {
    mode: 'development',
    devServer: {
        // proxy: {
        //     "/api": {
        //         target: "http://192.168.0.3:8000",
        //     },
        // },
        static: {
            directory: path.join(__dirname, "../docs"),
        },
        headers: {
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
        },
        https: true,
        client: {
            overlay: {
                errors: false,
                warnings: false,
            },
        },
    },
})
