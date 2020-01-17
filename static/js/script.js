$(function() {
  $('form').submit(function(e) {
    var spinner = $('#loader');
    spinner.show();
  });
});

function backButton(relative_path) {
    window.location.href= relative_path;
}