document.addEventListener('DOMContentLoaded', function() {
    var images = document.querySelectorAll('.gallery-img');
    var modal = document.getElementById('imageModal');
    var modalImg = document.getElementById('modalImg');
    var span = document.getElementsByClassName('close')[0];

    images.forEach(function(img) {
        img.onclick = function(e) {
            e.stopPropagation();
            modal.style.display = 'flex';
            modalImg.src = this.src;
        }
    });

    span.onclick = function() {
        modal.style.display = 'none';
    }

    modal.onclick = function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    }

    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            modal.style.display = 'none';
        }
    });
});
