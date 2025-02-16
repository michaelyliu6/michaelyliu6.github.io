// Initialize OpenSeadragon viewers for images
document.addEventListener('DOMContentLoaded', function() {
    // Find all images in the content
    const images = document.querySelectorAll('.post-content img:not(.nozoom)');
    
    images.forEach((img) => {
        // Create wrapper for the image and controls
        const wrapper = document.createElement('div');
        wrapper.className = 'image-zoom-wrapper';
        img.parentNode.insertBefore(wrapper, img);
        wrapper.appendChild(img);

        // Create zoom controls
        const controls = document.createElement('div');
        controls.className = 'zoom-controls';
        
        const zoomIn = document.createElement('button');
        zoomIn.className = 'zoom-button zoom-in';
        zoomIn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zm2.5-4h-2v2H9v-2H7V9h2V7h1v2h2v1z"/></svg>';
        
        const zoomOut = document.createElement('button');
        zoomOut.className = 'zoom-button zoom-out';
        zoomOut.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zM7 9h5v1H7z"/></svg>';
        
        controls.appendChild(zoomIn);
        controls.appendChild(zoomOut);
        wrapper.appendChild(controls);

        // Initialize zoom state
        let scale = 1;
        let panning = false;
        let pointX = 0;
        let pointY = 0;
        let start = { x: 0, y: 0 };

        // Get boundaries
        function getBoundaries() {
            const wrapperRect = wrapper.getBoundingClientRect();
            const imgRect = img.getBoundingClientRect();
            
            // Calculate the scaled dimensions
            const scaledWidth = imgRect.width / scale;
            const scaledHeight = imgRect.height / scale;
            
            // Calculate the maximum pan distances to keep image in view
            const maxPanX = Math.max(0, (scaledWidth * scale - wrapperRect.width) / 2);
            const maxPanY = Math.max(0, (scaledHeight * scale - wrapperRect.height) / 2);
            
            return {
                minX: -maxPanX,
                maxX: maxPanX,
                minY: -maxPanY,
                maxY: maxPanY
            };
        }

        // Constrain position within boundaries
        function constrainPosition() {
            if (scale <= 1) {
                pointX = 0;
                pointY = 0;
                return;
            }
            
            const bounds = getBoundaries();
            pointX = Math.max(bounds.minX, Math.min(bounds.maxX, pointX));
            pointY = Math.max(bounds.minY, Math.min(bounds.maxY, pointY));
        }

        // Zoom functions
        function setTransform() {
            constrainPosition();
            img.style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
        }

        zoomIn.addEventListener('click', function(e) {
            e.preventDefault();
            if (scale < 4) {
                const prevScale = scale;
                scale *= 1.2;
                // Adjust position to maintain relative view position
                pointX = (pointX * scale) / prevScale;
                pointY = (pointY * scale) / prevScale;
                setTransform();
            }
        });

        zoomOut.addEventListener('click', function(e) {
            e.preventDefault();
            const prevScale = scale;
            scale = Math.max(1, scale / 1.2);
            // Adjust position to maintain relative view position
            if (scale > 1) {
                pointX = (pointX * scale) / prevScale;
                pointY = (pointY * scale) / prevScale;
            } else {
                pointX = 0;
                pointY = 0;
            }
            setTransform();
        });

        // Pan functionality
        img.addEventListener('mousedown', function(e) {
            e.preventDefault();
            if (scale > 1) {
                panning = true;
                start = {
                    x: e.clientX - pointX,
                    y: e.clientY - pointY
                };
                img.style.cursor = 'grabbing';
            }
        });

        document.addEventListener('mousemove', function(e) {
            if (!panning) return;
            e.preventDefault();
            pointX = (e.clientX - start.x);
            pointY = (e.clientY - start.y);
            setTransform();
        });

        document.addEventListener('mouseup', function(e) {
            panning = false;
            img.style.cursor = 'grab';
        });

        // Reset zoom on double click
        img.addEventListener('dblclick', function(e) {
            e.preventDefault();
            scale = 1;
            pointX = 0;
            pointY = 0;
            setTransform();
        });

        // Prevent dragging image
        img.addEventListener('dragstart', function(e) {
            e.preventDefault();
        });

        // Handle mouse leaving the window
        document.addEventListener('mouseleave', function(e) {
            panning = false;
            img.style.cursor = 'grab';
        });

        // Initial setup
        setTransform();
    });
}); 