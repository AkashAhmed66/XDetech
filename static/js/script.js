document.addEventListener('DOMContentLoaded', function() {
    // File upload preview functionality
    const fileUpload = document.getElementById('file-upload');
    const filePreview = document.getElementById('file-preview');
    const previewImage = document.getElementById('preview-image');
    const fileName = document.getElementById('file-name');
    const removeFile = document.getElementById('remove-file');
    const modernUploadArea = document.querySelector('.modern-upload-area');
    
    if (fileUpload) {
        fileUpload.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                
                // Check if the file is an image
                if (!file.type.match('image.*')) {
                    alert('Please select an image file (JPEG, PNG)');
                    return;
                }
                
                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    fileName.textContent = file.name;
                    filePreview.classList.remove('d-none');
                    modernUploadArea.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        const uploadArea = document.querySelector('.modern-upload-area');
        if (uploadArea) {
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight() {
                uploadArea.classList.add('highlight');
            }
            
            function unhighlight() {
                uploadArea.classList.remove('highlight');
            }
            
            uploadArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                fileUpload.files = files;
                
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                fileUpload.dispatchEvent(event);
            }
        }
    }
    
    // Remove file preview
    if (removeFile) {
        removeFile.addEventListener('click', function() {
            fileUpload.value = '';
            filePreview.classList.add('d-none');
            modernUploadArea.style.display = 'flex';
        });
    }
    
    // Add highlight class to upload area on focus
    if (fileUpload) {
        fileUpload.addEventListener('focus', function() {
            modernUploadArea.classList.add('highlight');
        });
        
        fileUpload.addEventListener('blur', function() {
            modernUploadArea.classList.remove('highlight');
        });
    }
    
    // Add CSS class for drag-and-drop highlight
    const style = document.createElement('style');
    style.textContent = `
        .modern-upload-area.highlight {
            background-color: #e8f0fe;
            border-color: #4285f4;
            border-style: solid;
        }
    `;
    document.head.appendChild(style);
}); 