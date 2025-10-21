// Minimal JS for later UX improvements.
// Currently placeholder: image preview hook could be added here.

document.addEventListener("DOMContentLoaded", function () {
  // placeholder: attach image preview
  const fileInput = document.querySelector('input[type="file"]');
  if (!fileInput) return;

  fileInput.addEventListener("change", (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    // future: show preview in sidebar
    console.debug("Selected file:", file.name);
  });
});
