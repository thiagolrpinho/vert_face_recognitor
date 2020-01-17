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
    if (id_list.length > 0)
    {
        document.getElementById(id_list[0]).addEventListener('change',function(){
            //get the file name
            var label_id = 'labelGroupFile01'
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
            console.log(document)
            console.log(id_list.length)
            console.log(label_id)
            document.getElementById(label_id).textContent = fileName
        })
    }
    if (id_list.length > 1)
    {
        document.getElementById(id_list[1]).addEventListener('change',function(){
            //get the file name
            var label_id = 'labelGroupFile02'
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
            console.log(document)
            console.log(id_list.length)
            document.getElementById(label_id).textContent = fileName
        })
    }
}

$(document).ready(function(document) {
    if (window.location.pathname == '/reconhecimento_facial/')
    {
        inputLabel(['first_image','second_image'])
    } else if (window.location.pathname == '/renach/')
    {
        inputLabel(['first_image'])
    }
})