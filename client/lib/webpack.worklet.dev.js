const { merge } = require("webpack-merge");
const common = require("./webpack.worklet.common.js");

const worklet = merge(common, {
    mode: "development",
});
module.exports = [worklet];
