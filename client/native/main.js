const { app, dialog, BrowserWindow } = require('electron')
const parseArgs = require('electron-args');

const cli = parseArgs(`
    voice-changer-native-client

    Usage
      $ <command> <url>

    Options
      --help       show help
      --version    show version
      --url,-u  open client

    Examples
      $ voice-changer-native-client http://localhost:18888/
`, {
    alias: {
        u: 'url'
    },
    default: {
        url: "http://localhost:18888/"
    }
});

console.log(cli.flags);
console.log(cli.flags["url"]);

const url = cli.flags["url"]

const createWindow = () => {
    const win = new BrowserWindow({
        width: 800,
        height: 600
    })

    app.on('certificate-error', function (event, webContents, url, error, certificate, callback) {
        event.preventDefault();
        callback(true);
    });

    win.loadURL(url)
}

app.whenReady().then(() => {
    createWindow()
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})
