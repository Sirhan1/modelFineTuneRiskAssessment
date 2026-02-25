window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

function typesetMath() {
  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    window.MathJax.typesetPromise();
  }
}

function loadMathJax() {
  const cdnUrls = [
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
    "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js",
  ];

  const tryLoad = (index) => {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      typesetMath();
      return;
    }
    if (index >= cdnUrls.length) {
      console.error("MathJax failed to load from all configured CDNs.");
      return;
    }

    const script = document.createElement("script");
    script.src = cdnUrls[index];
    script.async = true;
    script.onload = () => typesetMath();
    script.onerror = () => tryLoad(index + 1);
    document.head.appendChild(script);
  };

  tryLoad(0);
}

if (window.document$ && typeof window.document$.subscribe === "function") {
  window.document$.subscribe(typesetMath);
}

window.addEventListener("load", loadMathJax);
