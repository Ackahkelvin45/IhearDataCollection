/*!

=========================================================
* Argon Dashboard Tailwind - v1.0.1
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-dashboard-tailwind
* Copyright 2022 Creative Tim (https://www.creative-tim.com)

* Coded by www.creative-tim.com

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/

// Use the global staticBaseUrl defined in the HTML template
var staticBaseUrl = window.staticBaseUrl || "/static/";  // Fallback to /static/ if not defined

// Modified loader functions that use Django's static tag
function loadJS(FILE_PATH, async) {
  let dynamicScript = document.createElement("script");
  dynamicScript.src = staticBaseUrl + FILE_PATH;
  dynamicScript.type = "text/javascript";
  dynamicScript.async = async;
  document.head.appendChild(dynamicScript);
}

function loadStylesheet(FILE_PATH) {
  let dynamicStylesheet = document.createElement("link");
  dynamicStylesheet.href = staticBaseUrl + FILE_PATH;
  dynamicStylesheet.rel = "stylesheet";
  document.head.appendChild(dynamicStylesheet);
}

// Conditional loading
if (document.querySelector("[slider]")) {
  loadJS("assets/js/carousel.js", true);
}

if (document.querySelector("nav [navbar-trigger]")) {
  loadJS("assets/js/navbar-collapse.js", true);
}

if (document.querySelector("[data-target='tooltip']")) {
  loadJS("assets/js/tooltips.js", true);
  loadStylesheet("assets/css/tooltips.css");
}

if (document.querySelector("[nav-pills]")) {
  loadJS("assets/js/nav-pills.js", true);
}

if (document.querySelector("[dropdown-trigger]")) {
  loadJS("assets/js/dropdown.js", true);
}

if (document.querySelector("[fixed-plugin]")) {
  loadJS("assets/js/fixed-plugin.js", true);
}

if (document.querySelector("[navbar-main]") || document.querySelector("[navbar-profile]")) {
  if(document.querySelector("[navbar-main]")){
    loadJS("assets/js/navbar-sticky.js", true);
  }
  if (document.querySelector("aside")) {
    loadJS("assets/js/sidenav-burger.js", true);
  }
}

if (document.querySelector("canvas")) {
  loadJS("assets/js/charts.js", true);
}

if (document.querySelector(".github-button")) {
  loadJS("https://buttons.github.io/buttons.js", true);
}
