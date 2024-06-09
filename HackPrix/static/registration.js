document.getElementById('registration-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const name = document.getElementById('name').value;
    const bp = document.getElementById('bp').value;
    const heartRate = document.getElementById('heart-rate').value;
    const sugarLevel = document.getElementById('sugar-level').value;

    const userInfo = {
        name: name,
        bp: bp,
        heartRate: heartRate,
        sugarLevel: sugarLevel
    };

    localStorage.setItem('userInfo', JSON.stringify(userInfo));

    window.location.href = 'main.html';
});
