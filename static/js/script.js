$(function() {
  $('form').submit(function(e) {
    var spinner = $('#loader');
    spinner.show();
  });
});

function backButton(relative_path) {
    window.location.href= relative_path;
}

function inputLabel(id_list){
  for (var i = 0; i < id_list.length; i++) {
    
    document.getElementById(id_list[i]).addEventListener('change',function(){
      //get the file name
      var fullPath = this.value;
      var fileName = "arquivo_sem_nome"
      if (fullPath) {
        var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
        fileName = fullPath.substring(startIndex);
        if (fileName.indexOf('\\') === 0 || fileName.indexOf('/') === 0) {
          fileName = fileName.substring(1);
        }
      }
      //replace the "Choose a file" label
      document.getElementById('labelGroupFile0' + (i+1).toString()).textContent = fileName
    })
  }
}

$(document).ready(function() {
    inputLabel(['first_image','second_image'])
})