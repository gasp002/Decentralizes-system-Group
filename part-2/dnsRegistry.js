const express = require('express');
const app = express();
const port = 3002; // Use a different port

app.get('/getServer', (req, res) => {
  res.json({ code: 200, server: "localhost:3000" }); // Change the port to your Hello World server's port
});

app.listen(port, () => {
  console.log(`DNS Registry listening at http://localhost:${port}`);
});