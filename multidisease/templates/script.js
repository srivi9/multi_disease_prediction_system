document.addEventListener("DOMContentLoaded", function () {
    const slides = document.querySelector(".slides");
    let index = 0;

    function nextSlide() {
        index++;
        if (index > 2) index = 0; // Reset to first image after last one
        slides.style.transform = `translateX(-${index * 100}vw)`;
    }

    setInterval(nextSlide, 10000); // Change image every 4 seconds
});
