const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("quantDesktop", {
  appMode: "desktop",
});
