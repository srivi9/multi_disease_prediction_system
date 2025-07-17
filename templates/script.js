document.addEventListener("DOMContentLoaded", function () {
    const slidesContainer = document.querySelector(".slides");
    const slides = document.querySelectorAll(".slide");
    let index = 0;

    function nextSlide() {
        index++;
        if (index >= slides.length) {
            index = 0; // Reset to first image
        }
        slidesContainer.style.transform = `translateX(-${index * 100}%)`;
    }

    setInterval(nextSlide, 5000); // Change slide every 5 seconds

    // Fade-in Effect for Elements
    function fadeInOnScroll() {
        const fadeInElements = document.querySelectorAll('.fade-in');
        
        fadeInElements.forEach(element => {
            const rect = element.getBoundingClientRect();
            if (rect.top >= 0 && rect.bottom <= window.innerHeight) {
                element.classList.add('visible');
            }
        });
    }

    window.addEventListener('scroll', fadeInOnScroll);
    fadeInOnScroll(); // Ensure fade-in effect applies immediately on load
});
