// Funcionalidad básica para el menú hamburguesa en dispositivos móviles
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const caso1Link = document.querySelector('/caso1');
    const caso2Link = document.querySelector('/caso2');
    const caso3Link = document.querySelector('/caso1');
    const caso4Link = document.querySelector('/caso4');
    
    if (hamburger) {
        hamburger.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            caso1Link.classList.toggle('active');
            caso2Link.classList.toggle('active');
            caso3Link.classList.toggle('active');
            caso4Link.classList.toggle('active');
        });
    }
    
    // Smooth scrolling para los enlaces del submenú
    document.querySelectorAll('.submenu-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                // Cerrar el menú móvil si está abierto
                if (navMenu.classList.contains('active')) {
                    navMenu.classList.remove('active');
                }
            }
        });
    });
});