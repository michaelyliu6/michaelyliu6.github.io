// Initialize OpenSeadragon viewers for images
document.addEventListener('DOMContentLoaded', function() {
    // Find all images in the content
    const images = document.querySelectorAll('.post-content img:not(.nozoom)');
    
    // Create the fullscreen overlay - structured with a dedicated background for click handling
    const fullscreenOverlay = document.createElement('div');
    fullscreenOverlay.className = 'fullscreen-overlay';
    fullscreenOverlay.style.cssText = 'display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background-color:rgba(0,0,0,0.9); z-index:9999; cursor:pointer;';
    
    // Create a container for the image that won't receive background clicks
    const fullscreenContainer = document.createElement('div');
    fullscreenContainer.className = 'fullscreen-container';
    fullscreenContainer.style.cssText = 'position:absolute; top:0; left:0; width:100%; height:100%; display:flex; justify-content:center; align-items:center; pointer-events:none;';
    
    // Create the image element
    const fullscreenImage = document.createElement('img');
    fullscreenImage.className = 'fullscreen-image';
    fullscreenImage.style.cssText = 'max-width:95vw; max-height:95vh; object-fit:contain; cursor:grab; pointer-events:auto;';
    
    // Create the close button
    const fullscreenClose = document.createElement('button');
    fullscreenClose.className = 'fullscreen-close';
    fullscreenClose.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
    fullscreenClose.style.cssText = 'position:absolute; top:20px; right:20px; background:rgba(0,0,0,0.5); border:none; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center; cursor:pointer; color:white; z-index:10000; pointer-events:auto;';
    
    // Assemble the fullscreen elements
    fullscreenContainer.appendChild(fullscreenImage);
    fullscreenContainer.appendChild(fullscreenClose);
    fullscreenOverlay.appendChild(fullscreenContainer);
    document.body.appendChild(fullscreenOverlay);
    
    // Fullscreen state
    let fsScale = 1;
    let fsPanning = false;
    let fsPointX = 0;
    let fsPointY = 0;
    let fsStart = { x: 0, y: 0 };
    
    // Fullscreen functions
    function closeFullscreen() {
        fullscreenOverlay.style.display = 'none';
        document.body.style.overflow = '';
        fsScale = 1;
        fsPointX = 0;
        fsPointY = 0;
        setFullscreenTransform();
    }
    
    function setFullscreenTransform() {
        fullscreenImage.style.transform = `translate(${fsPointX}px, ${fsPointY}px) scale(${fsScale})`;
    }
    
    // Fullscreen event listeners
    fullscreenClose.addEventListener('click', function(e) {
        e.stopPropagation();
        closeFullscreen();
    });
    
    // Click overlay background to close
    fullscreenOverlay.addEventListener('click', function(e) {
        // If clicked directly on the overlay (not the image or controls)
        if (e.target === fullscreenOverlay) {
            closeFullscreen();
        }
    });
    
    fullscreenImage.addEventListener('click', function(e) {
        e.stopPropagation(); // Prevent clicks on the image from closing the overlay
    });
    
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && fullscreenOverlay.style.display === 'flex') {
            closeFullscreen();
        }
    });
    
    window.addEventListener('resize', function() {
        if (fullscreenOverlay.style.display === 'flex') {
            fsScale = 1;
            fsPointX = 0;
            fsPointY = 0;
            setFullscreenTransform();
        }
    });
    
    // Fullscreen mouse events
    fullscreenImage.addEventListener('mousedown', function(e) {
        e.stopPropagation();
        if (fsScale > 1) {
            fsPanning = true;
            fsStart = {
                x: e.clientX - fsPointX,
                y: e.clientY - fsPointY
            };
            fullscreenImage.style.cursor = 'grabbing';
        }
    });
    
    document.addEventListener('mousemove', function(e) {
        if (fsPanning) {
            e.preventDefault();
            fsPointX = (e.clientX - fsStart.x);
            fsPointY = (e.clientY - fsStart.y);
            setFullscreenTransform();
        }
    });
    
    document.addEventListener('mouseup', function() {
        fsPanning = false;
        fullscreenImage.style.cursor = 'grab';
    });
    
    fullscreenImage.addEventListener('wheel', function(e) {
        e.stopPropagation();
        e.preventDefault();
        const delta = e.deltaY > 0 ? -1 : 1;
        const prevScale = fsScale;
        
        if (delta > 0 && fsScale < 4) {
            fsScale *= 1.2;
        } else if (delta < 0) {
            fsScale = Math.max(1, fsScale / 1.2);
        }
        
        if (fsScale > 1) {
            fsPointX = (fsPointX * fsScale) / prevScale;
            fsPointY = (fsPointY * fsScale) / prevScale;
        } else {
            fsPointX = 0;
            fsPointY = 0;
        }
        
        setFullscreenTransform();
    });
    
    fullscreenImage.addEventListener('dblclick', function(e) {
        e.stopPropagation();
        e.preventDefault();
        fsScale = 1;
        fsPointX = 0;
        fsPointY = 0;
        setFullscreenTransform();
    });
    
    // Process each image
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
        
        // Add fullscreen button
        const fullscreenBtn = document.createElement('button');
        fullscreenBtn.className = 'zoom-button fullscreen';
        fullscreenBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
        
        controls.appendChild(zoomIn);
        controls.appendChild(zoomOut);
        controls.appendChild(fullscreenBtn);
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

        // Fullscreen button click handler
        fullscreenBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            // Set the source of the fullscreen image
            fullscreenImage.src = img.src;
            fullscreenImage.alt = img.alt || '';
            
            // Display the fullscreen overlay
            fullscreenOverlay.style.display = 'flex';
            
            // Prevent scrolling on the body
            document.body.style.overflow = 'hidden';
            
            // Reset fullscreen state
            fsScale = 1;
            fsPointX = 0;
            fsPointY = 0;
            setFullscreenTransform();
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