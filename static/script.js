document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const status = document.getElementById('status');
    const videoContainer = document.getElementById('video-container');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '#e0e6e8';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
        const file = e.dataTransfer.files[0];
        if (file.name.endsWith('.r3d')) {
            uploadFile(file);
        } else {
            status.textContent = 'Please upload a .r3d file.';
        }
    });

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload', true);

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                progressBar.style.width = percentComplete + '%';
            }
        };

        xhr.onload = function() {
            if (xhr.status === 200) {
                status.textContent = 'File uploaded successfully. Processing...';
                checkVideoStatus(xhr.responseText, file.name);
            } else {
                status.textContent = 'Upload failed. Please try again.';
                progressContainer.style.display = 'none';
            }
        };

        xhr.send(formData);
    }

    function checkVideoStatus(taskId, fileName) {
        const statusCheck = setInterval(() => {
            fetch(`/status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'completed') {
                        clearInterval(statusCheck);
                        status.textContent = 'Video processing completed.';
                        progressContainer.style.display = 'none';
                        displayVideo(data.video_url, fileName);
                    } else if (data.status === 'failed') {
                        clearInterval(statusCheck);
                        status.textContent = 'Video processing failed. Please try again.';
                        progressContainer.style.display = 'none';
                    }
                });
        }, 2000);
    }

    function displayVideo(videoUrl, fileName) {
        const videoWrapper = document.createElement('div');
        videoWrapper.className = 'video-wrapper';

        const videoTitle = document.createElement('div');
        videoTitle.className = 'video-title';
        videoTitle.textContent = fileName;

        const videoElement = document.createElement('video');
        videoElement.src = videoUrl;
        videoElement.controls = true;

        videoWrapper.appendChild(videoTitle);
        videoWrapper.appendChild(videoElement);
        videoContainer.prepend(videoWrapper);
    }
});
