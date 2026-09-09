// Query text stays in the browser. Only an explicit result choice sends an event.
export function rankedContacts(data, query) {
  const q = Array.from(query.trim()).map(c => data.folds[c] ?? c.toLowerCase()).join("");
  if (!q) return [];
  return data.contacts.map((contact, order) => {
    const [first, last, full, company, email] = contact.fields;
    const matches = [first.startsWith(q), last.startsWith(q), full.startsWith(q),
      first.includes(q), last.includes(q), full.includes(q),
      company.startsWith(q), company.includes(q), email.startsWith(q), email.includes(q)];
    return {contact, order, priority: matches.indexOf(true)};
  }).filter(item => item.priority >= 0)
    .sort((a, b) => a.priority - b.priority || a.order - b.order).map(item => item.contact);
}

export default function({parentElement, data, setTriggerValue}) {
  const input = parentElement.querySelector("input");
  const menu = parentElement.querySelector('[role="listbox"]');
  if (input.dataset.selectedId !== data.selected_id) {
    input.value = "";
    input.dataset.selectedId = data.selected_id;
  }
  let results = [];
  let active = -1;
  function close() {
    menu.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }
  function choose(contact) {
    input.value = "";
    close();
    setTriggerValue("selected", contact.id);
  }
  function show() {
    results = rankedContacts(data, input.value);
    active = -1;
    menu.replaceChildren();
    input.removeAttribute("aria-activedescendant");
    if (!input.value.trim()) return close();
    menu.hidden = false;
    input.setAttribute("aria-expanded", "true");
    if (!results.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.setAttribute("role", "status");
      empty.textContent = "No matching client contacts.";
      menu.append(empty);
    }
    results.forEach((contact, index) => {
      const option = document.createElement("div");
      option.id = `contact-option-${index}`;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.textContent = contact.label;
      option.onmousedown = event => event.preventDefault();
      option.onclick = () => choose(contact);
      menu.append(option);
    });
  }
  input.oninput = show;
  input.onfocus = show;
  input.onblur = close;
  input.onkeydown = event => {
    if (event.key === "Escape") return close();
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (menu.hidden) show();
      if (!results.length) return;
      active = (active + (event.key === "ArrowDown" ? 1 : -1) + results.length) % results.length;
      Array.from(menu.children).forEach((option, index) => option.setAttribute("aria-selected", String(index === active)));
      input.setAttribute("aria-activedescendant", menu.children[active].id);
      menu.children[active].scrollIntoView({block: "nearest"});
    } else if (event.key === "Enter" && !menu.hidden && active >= 0) {
      event.preventDefault();
      choose(results[active]);
    }
  };
  close();
}
