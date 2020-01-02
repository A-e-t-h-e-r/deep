const path = require('path');

module.exports = {
    entry: './src/index.js',
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist')
    },
    devtool: 'inline-source-map',
    devServer: {
        contentBase: path.resolve(__dirname, 'dist'), //设置基本目录结构,本地服务器所加载的页面所在的目录
        host: 'localhost',//服务器的IP地址，可以使用IP也可以使用localhost
        inline: true,//实时刷新
        compress: true,//服务端压缩是否开启，一般设置为开启，
        historyApiFallback: true,//不跳转
        port: 1818 //配置服务端口号，建议不使用80，很容易被占用，这里使用了1818
    }
};
