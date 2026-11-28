// Chat UI: sends messages to /api/chat and renders the replies.
(function () {
  var messagesEl = document.getElementById("messages");
  var chipsEl = document.getElementById("chips");
  var form = document.getElementById("chat-form");
  var input = document.getElementById("message-input");
  var sendBtn = document.getElementById("send-btn");
  var resetBtn = document.getElementById("reset-btn");

  var sessionId = (window.crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : "session-" + Date.now() + "-" + Math.floor(Math.random() * 1e9);
  var typingEl = null;

  function scrollDown() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addBubble(text, who) {
    var bubble = document.createElement("div");
    bubble.className = "bubble " + who;
    bubble.textContent = text;
    messagesEl.appendChild(bubble);
    scrollDown();
    return bubble;
  }

  function showTyping() {
    typingEl = addBubble("•••", "bot typing");
  }

  function hideTyping() {
    if (typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }

  function showChips(chips) {
    chipsEl.innerHTML = "";
    (chips || []).forEach(function (chip) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "chip";
      button.textContent = chip;
      button.addEventListener("click", function () {
        sendMessage(chip);
      });
      chipsEl.appendChild(button);
    });
  }

  function sendMessage(text) {
    text = (text || "").trim();
    if (!text) {
      return;
    }
    addBubble(text, "user");
    showChips([]);
    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;
    showTyping();

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    })
      .then(function (res) { return res.json().then(function (body) {
        return { ok: res.ok, body: body };
      }); })
      .then(function (result) {
        hideTyping();
        if (result.ok) {
          addBubble(result.body.reply, "bot");
          showChips(result.body.quick_replies);
        } else {
          addBubble(result.body.error || "Something went wrong, please try again.", "bot");
        }
      })
      .catch(function () {
        hideTyping();
        addBubble("Could not reach the server. Please check your connection and try again.", "bot");
      })
      .finally(function () {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
      });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    sendMessage(input.value);
  });

  resetBtn.addEventListener("click", function () {
    fetch("/api/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).finally(function () {
      messagesEl.innerHTML = "";
      showChips([]);
      addBubble("Hello! I can help with admissions, courses, fees, timetable, tests and results. What would you like to know?", "bot");
      showChips(["Admissions", "Fees", "Courses", "Timetable", "Tests", "Results"]);
      input.focus();
    });
  });

  addBubble("Hello! I can help with admissions, courses, fees, timetable, tests and results. What would you like to know?", "bot");
  showChips(["Admissions", "Fees", "Courses", "Timetable", "Tests", "Results"]);
  input.focus();
})();
