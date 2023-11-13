const { merge } = require("webpack-merge");
const common = require("./webpack_web.common.js");

module.exports = merge(common, {
    mode: "production",
});
