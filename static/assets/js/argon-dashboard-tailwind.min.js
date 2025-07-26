/*!
=========================================================
* Argon Dashboard Tailwind - v1.0.1
=========================================================
* Product Page: https://www.creative-tim.com/product/argon-dashboard-tailwind
* Copyright 2022 Creative Tim (https://www.creative-tim.com)
* Coded by www.creative-tim.com
=========================================================
*/

// Track loaded scripts to prevent duplicates
var loadedScripts = {};
var loadedStylesheets = {};

// Use the global staticBaseUrl defined in the HTML template
var staticBaseUrl = window.staticBaseUrl || "/static/";

function loadJS(FILE_PATH, async) {
  if (loadedScripts[FILE_PATH]) return;

  let dynamicScript = document.createElement("script");
  dynamicScript.src = staticBaseUrl + FILE_PATH;
  dynamicScript.type = "text/javascript";
  dynamicScript.async = async;
  document.head.appendChild(dynamicScript);

  loadedScripts[FILE_PATH] = true;
}

function loadStylesheet(FILE_PATH) {
  if (loadedStylesheets[FILE_PATH]) return;

  let dynamicStylesheet = document.createElement("link");
  dynamicStylesheet.href = staticBaseUrl + FILE_PATH;
  dynamicStylesheet.rel = "stylesheet";
  document.head.appendChild(dynamicStylesheet);

  loadedStylesheets[FILE_PATH] = true;
}

// Load essential files first
loadStylesheet("assets/css/perfect-scrollbar.css");
loadJS("assets/js/perfect-scrollbar.js", true);

// Conditional loading
if (document.querySelector("[slider]") && !loadedScripts["assets/js/carousel.js"]) {
  loadJS("assets/js/carousel.js", true);
}

if (document.querySelector("nav [navbar-trigger]") && !loadedScripts["assets/js/navbar-collapse.js"]) {
  loadJS("assets/js/navbar-collapse.js", true);
}

if (document.querySelector("[data-target='tooltip']")) {
  if (!loadedScripts["assets/js/tooltips.js"]) loadJS("assets/js/tooltips.js", true);
  if (!loadedStylesheets["assets/css/tooltips.css"]) loadStylesheet("assets/css/tooltips.css");
}

// ... (keep other conditional loading blocks the same, but add the !loadedScripts checks)
