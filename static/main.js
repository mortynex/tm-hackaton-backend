  const imageUpload = document.getElementById('imageUpload');
  const seedInput = document.getElementById('seed');
  const motionIdInput = document.getElementById('motionId');
  const fpsInput = document.getElementById('fps');
  const generateButton = document.getElementById('generateButton');
  const generatedVideo = document.getElementById('generatedVideo');
  const seedDisplay = document.getElementById('seedDisplay');

  generateButton.addEventListener('click', async (event) => {
    event.preventDefault(); // Prevent default form submission
    const file = imageUpload.files[0];
    const reader = new FileReader();

    reader.onload = async (event) => {
      const imageData = event.target.result; // Base64 encoded image
      console.log('imageData', imageData);
      const response = await fetch('generate_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageData: imageData,
          seed: seedInput.value,
          motionId: motionIdInput.value,
          fps: fpsInput.value
        }) 
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data);
        generatedVideo.src = 'data:video/mp4;base64,' + data.video; 
        seedDisplay.textContent = 'Seed: ' + data.seed;
      } else {
        alert('Error generating video');
      }
    }

    reader.readAsDataURL(file); 
  });
