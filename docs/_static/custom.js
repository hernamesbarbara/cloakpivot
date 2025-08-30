// Custom JavaScript for CloakPivot documentation

document.addEventListener('DOMContentLoaded', function() {
    
    // Add copy buttons to code blocks
    addCopyButtons();
    
    // Add CLI command highlighting
    highlightCliCommands();
    
    // Add collapsible sections
    addCollapsibleSections();
    
    // Add version selector functionality
    addVersionSelector();
});

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('div.highlight pre');
    
    codeBlocks.forEach(function(block) {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.title = 'Copy to clipboard';
        
        button.addEventListener('click', function() {
            const text = block.textContent;
            navigator.clipboard.writeText(text).then(function() {
                button.textContent = 'Copied!';
                button.classList.add('copied');
                
                setTimeout(function() {
                    button.textContent = 'Copy';
                    button.classList.remove('copied');
                }, 2000);
            }).catch(function(err) {
                console.error('Failed to copy text: ', err);
                button.textContent = 'Failed';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        const wrapper = block.parentElement;
        wrapper.style.position = 'relative';
        wrapper.appendChild(button);
    });
}

function highlightCliCommands() {
    const codeBlocks = document.querySelectorAll('div.highlight pre');
    
    codeBlocks.forEach(function(block) {
        const text = block.textContent;
        
        // Check if this looks like a CLI command
        if (text.includes('cloakpivot ') || text.match(/^\$ /m)) {
            block.classList.add('cli-command');
            
            // Apply syntax highlighting for CLI commands
            const lines = text.split('\n');
            let highlightedHtml = '';
            
            lines.forEach(function(line) {
                if (line.startsWith('$ ') || line.startsWith('cloakpivot ')) {
                    line = line.replace(/^\$ /, '<span class="prompt">$ </span>');
                    line = line.replace(/^cloakpivot /, '<span class="command">cloakpivot </span>');
                    line = line.replace(/(--\w+)/g, '<span class="option">$1</span>');
                    line = line.replace(/(\w+\.(json|yaml|pdf|html))/g, '<span class="argument">$1</span>');
                }
                highlightedHtml += line + '\n';
            });
            
            // Only apply if we found CLI patterns
            if (highlightedHtml !== text) {
                block.innerHTML = highlightedHtml;
            }
        }
    });
}

function addCollapsibleSections() {
    const headers = document.querySelectorAll('h3, h4');
    
    headers.forEach(function(header) {
        // Only make certain sections collapsible
        const collapsibleClasses = ['api-section', 'example-section', 'advanced-section'];
        
        if (collapsibleClasses.some(cls => header.classList.contains(cls))) {
            header.classList.add('collapsible');
            header.style.cursor = 'pointer';
            
            // Add toggle icon
            const icon = document.createElement('span');
            icon.className = 'toggle-icon';
            icon.textContent = '▼';
            icon.style.marginRight = '8px';
            icon.style.fontSize = '0.8em';
            header.insertBefore(icon, header.firstChild);
            
            // Find content to collapse
            let content = [];
            let nextElement = header.nextElementSibling;
            
            while (nextElement && !nextElement.matches('h1, h2, h3, h4, h5, h6')) {
                content.push(nextElement);
                nextElement = nextElement.nextElementSibling;
            }
            
            // Add click handler
            header.addEventListener('click', function() {
                const isCollapsed = icon.textContent === '▶';
                
                content.forEach(function(element) {
                    element.style.display = isCollapsed ? 'block' : 'none';
                });
                
                icon.textContent = isCollapsed ? '▼' : '▶';
            });
        }
    });
}

function addVersionSelector() {
    const versionInfo = document.querySelector('.version-info');
    
    if (versionInfo) {
        // Add version selector dropdown (placeholder for future implementation)
        const selector = document.createElement('select');
        selector.className = 'version-selector';
        
        const currentVersion = document.createElement('option');
        currentVersion.value = 'latest';
        currentVersion.textContent = 'v0.1.0 (latest)';
        currentVersion.selected = true;
        selector.appendChild(currentVersion);
        
        // Future versions would be added here dynamically
        
        selector.addEventListener('change', function() {
            // Future: Navigate to different version documentation
            console.log('Version changed to:', selector.value);
        });
        
        versionInfo.appendChild(selector);
    }
}

// Add CSS for copy buttons
const style = document.createElement('style');
style.textContent = `
.copy-button {
    position: absolute;
    top: 8px;
    right: 8px;
    background: #2980B9;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.3s;
}

div.highlight:hover .copy-button {
    opacity: 1;
}

.copy-button:hover {
    background: #3498DB;
}

.copy-button.copied {
    background: #27AE60;
}

.collapsible:hover {
    color: #2980B9;
}

.toggle-icon {
    transition: transform 0.2s;
}

.version-selector {
    float: right;
    padding: 4px 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    font-size: 12px;
}
`;
document.head.appendChild(style);