(function () {
  // Lightweight DOM helpers
  function qs(sel) {
    return document.querySelector(sel);
  }

  // Image preview
  const input = qs('input[type="file"][name="image"]');
  const preview = qs("#preview");

  if (input && preview) {
    input.addEventListener("change", function () {
      const file = this.files && this.files[0];
      if (!file) {
        preview.classList.add("d-none");
        preview.removeAttribute("src");
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove("d-none");
      };
      reader.readAsDataURL(file);
    });
  }
})();
