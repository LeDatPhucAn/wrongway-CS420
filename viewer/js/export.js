/**
 * Board export utilities: PNG snapshot and optional video clip.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};

  class ExportManager {
    constructor(boardRenderer, controller) {
      this.renderer = boardRenderer;
      this.controller = controller;
    }

    exportPNG() {
      const svg = this.renderer.svg;
      if (!svg) return;

      const serializer = new XMLSerializer();
      let source = serializer.serializeToString(svg);

      // Add namespaces if missing
      if (!source.match(/^<svg[^>]+xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)) {
        source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
      }

      const svgBlob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = svg.viewBox.baseVal.width || 602;
        canvas.height = svg.viewBox.baseVal.height || 602;
        const ctx = canvas.getContext('2d');

        // Draw background
        const isDark = document.documentElement.classList.contains('dark');
        ctx.fillStyle = isDark ? '#070c16' : '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);

        canvas.toBlob((blob) => {
          if (!blob) return;
          const a = document.createElement('a');
          const turn = this.controller.currentIndex;
          a.download = `wrongway_turn_${turn}.png`;
          a.href = URL.createObjectURL(blob);
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        }, 'image/png');
      };
      img.src = url;
    }
  }

  window.WW.ExportManager = ExportManager;
})(window);
