/* MKAngel — Ladbrokes Treble stake bookmarklet
 *
 * What it does
 * ------------
 * Runs inside YOUR logged-in Ladbrokes Chrome tab, reads the betslip DOM,
 * finds the "Treble" (3-fold) row, and fills its stake input with £1.00.
 * You still tap "Place Bets" yourself -- the bookmarklet intentionally
 * never touches the confirm button.
 *
 * Credentials / safety
 * --------------------
 * - Never leaves your browser. No network request is made to anywhere.
 * - No data is sent to MKAngel, Anthropic, or anyone else.
 * - No cookies, no session tokens, no API calls. 100% local DOM tweak.
 * - Source is human-readable below; paste the one-liner into Chrome
 *   only after you've read what it does.
 *
 * How to install (once)
 * ---------------------
 * 1. In Chrome, show your bookmarks bar  (Ctrl+Shift+B / Cmd+Shift+B)
 * 2. Right-click the bar -> "Add page..."  (or "Add bookmark")
 * 3. Name:  "MKA Treble £1"
 * 4. URL:   paste the one-line version at the bottom of this file
 *          (the line that starts with  javascript:  )
 * 5. Save.
 *
 * How to use (every time)
 * -----------------------
 * 1. Go to sports.ladbrokes.com (or the Ladbrokes site), log in.
 * 2. Add your three Premier League selections to the betslip:
 *       - Burnley to win vs Brighton      (Sat 15:00)
 *       - Fulham to win at Liverpool      (Sat 17:30)
 *       - Aston Villa to win at Forest    (Sun 14:00)
 *    by clicking the three individual prices.
 * 3. Open the betslip (click the slip icon, or it slides out).
 * 4. Scroll until you see "Multiples" and a "Treble" row.
 * 5. Click the "MKA Treble £1" bookmark. A green toast in the top-right
 *    should confirm "Treble stake set to £1.00". The stake input will
 *    flash green for a few seconds.
 * 6. Review the betslip. Returns should show ~£50.
 * 7. Tap "Place Bets" yourself.  The bookmarklet will NOT do this.
 *
 * If it says "could not find the Treble stake input"
 * --------------------------------------------------
 * - Make sure all 3 selections are actually added (the slip icon should
 *   show a 3).
 * - Make sure the "Multiples" section is expanded/visible.
 * - If Ladbrokes has restructured their betslip DOM, come back to this
 *   file and I'll fix the selectors.
 */

(() => {
  const STAKE = '1.00';

  // Accept Treble / 3-fold / threefold variants in the row label.
  const LABEL_RE = /^\s*(treble|3\s*-?\s*fold|threefold)\s*$/i;

  // Search both the top document and any same-origin iframes -- betslips
  // are sometimes rendered in a frame.
  const docs = [document];
  for (const f of document.querySelectorAll('iframe')) {
    try { if (f.contentDocument) docs.push(f.contentDocument); } catch (e) {}
  }

  // Find every element whose own text looks like a Treble row label.
  const findTrebleLabels = (doc) => {
    const out = [];
    const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT);
    let n;
    while ((n = walker.nextNode())) {
      if (n.childNodes.length === 1 && n.childNodes[0].nodeType === 3) {
        const t = (n.textContent || '').trim();
        if (LABEL_RE.test(t) || (t.length < 40 && /\btreble\b/i.test(t))) {
          out.push(n);
        }
      }
    }
    return out;
  };

  // From a label element, walk up the DOM and look for a nearby stake
  // input. We prefer inputs whose attributes clearly name them as stake
  // fields, then fall back to any numeric/decimal input in the same
  // subtree.
  const findStakeInput = (labelEl) => {
    let parent = labelEl;
    for (let hop = 0; hop < 10 && parent; hop++) {
      const inputs = parent.querySelectorAll('input');
      // Pass 1: named stake inputs
      for (const input of inputs) {
        if (input.readOnly || input.disabled) continue;
        const attrs = [
          input.name, input.id, input.placeholder,
          input.getAttribute('aria-label') || '',
          input.getAttribute('data-test') || '',
          input.getAttribute('data-testid') || '',
        ].join(' ').toLowerCase();
        if (/stake|amount|wager|\bbet\b/.test(attrs)) return input;
      }
      // Pass 2: numeric / decimal inputs
      for (const input of inputs) {
        if (input.readOnly || input.disabled) continue;
        if (input.type === 'number' ||
            input.inputMode === 'decimal' ||
            input.inputMode === 'numeric') {
          return input;
        }
      }
      // Pass 3: single plain text input in this scope
      const textInputs = Array.from(inputs).filter(i =>
        ['', 'text', 'tel'].includes(i.type) && !i.readOnly && !i.disabled
      );
      if (textInputs.length === 1) return textInputs[0];
      parent = parent.parentElement;
    }
    return null;
  };

  // Find the first (doc, input) pair that works.
  let target = null, targetDoc = null;
  for (const doc of docs) {
    for (const label of findTrebleLabels(doc)) {
      const input = findStakeInput(label);
      if (input) { target = input; targetDoc = doc; break; }
    }
    if (target) break;
  }

  if (!target) {
    alert('MKAngel bookmarklet:\n\n'
        + 'Could not find the Treble stake input on this page.\n\n'
        + 'Checklist:\n'
        + ' - Is the betslip actually open?\n'
        + ' - Do you have all 3 selections added (slip count = 3)?\n'
        + ' - Is the "Multiples" / "Treble" section visible?\n\n'
        + 'If all of those are true, Ladbrokes may have updated their\n'
        + 'DOM. Paste this alert back to MKAngel and I will fix it.');
    return;
  }

  // React-safe value setter: frameworks like React intercept direct
  // .value= assignment on controlled inputs, so we set via the native
  // prototype and then dispatch both 'input' and 'change' events so
  // whatever state container is listening picks up the new value.
  const proto = targetDoc.defaultView.HTMLInputElement.prototype;
  const nativeSet = Object.getOwnPropertyDescriptor(proto, 'value').set;
  nativeSet.call(target, STAKE);
  target.dispatchEvent(new Event('input',  { bubbles: true }));
  target.dispatchEvent(new Event('change', { bubbles: true }));
  target.focus();
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });

  // Outline the input so it's obvious which field got filled.
  const prevOutline = target.style.outline;
  const prevOffset  = target.style.outlineOffset;
  target.style.outline = '4px solid #00e676';
  target.style.outlineOffset = '2px';
  setTimeout(() => {
    target.style.outline = prevOutline || '';
    target.style.outlineOffset = prevOffset || '';
  }, 4000);

  // Toast-style confirmation in the top-right of the page.
  try {
    const toast = targetDoc.createElement('div');
    toast.textContent = 'MKAngel: Treble stake set to £' + STAKE
                      + '. Review the slip, then tap Place Bets yourself.';
    toast.style.cssText = [
      'position:fixed', 'top:20px', 'right:20px',
      'background:#0a7f3f', 'color:#fff',
      'padding:14px 18px', 'border-radius:8px',
      'font:14px/1.4 system-ui,-apple-system,sans-serif',
      'z-index:2147483647', 'max-width:320px',
      'box-shadow:0 6px 20px rgba(0,0,0,.35)',
    ].join(';');
    targetDoc.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  } catch (e) {}
})();

/* -----------------------------------------------------------------------
 * ONE-LINE BOOKMARKLET (copy everything from `javascript:` to the final `;`)
 * -----------------------------------------------------------------------
 *
 * javascript:(()=>{const S='1.00';const R=/^\s*(treble|3\s*-?\s*fold|threefold)\s*$/i;const D=[document];for(const f of document.querySelectorAll('iframe')){try{if(f.contentDocument)D.push(f.contentDocument)}catch(e){}}const F=d=>{const o=[];const w=d.createTreeWalker(d.body,NodeFilter.SHOW_ELEMENT);let n;while(n=w.nextNode()){if(n.childNodes.length===1&&n.childNodes[0].nodeType===3){const t=(n.textContent||'').trim();if(R.test(t)||(t.length<40&&/\btreble\b/i.test(t)))o.push(n)}}return o};const G=e=>{let p=e;for(let h=0;h<10&&p;h++){const I=p.querySelectorAll('input');for(const x of I){if(x.readOnly||x.disabled)continue;const a=[x.name,x.id,x.placeholder,x.getAttribute('aria-label')||'',x.getAttribute('data-test')||'',x.getAttribute('data-testid')||''].join(' ').toLowerCase();if(/stake|amount|wager|\bbet\b/.test(a))return x}for(const x of I){if(x.readOnly||x.disabled)continue;if(x.type==='number'||x.inputMode==='decimal'||x.inputMode==='numeric')return x}const T=Array.from(I).filter(i=>['','text','tel'].includes(i.type)&&!i.readOnly&&!i.disabled);if(T.length===1)return T[0];p=p.parentElement}return null};let t=null,td=null;for(const d of D){for(const l of F(d)){const i=G(l);if(i){t=i;td=d;break}}if(t)break}if(!t){alert('MKAngel: could not find the Treble stake input. Make sure all 3 selections are added and the Multiples section is visible.');return}const P=td.defaultView.HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(P,'value').set.call(t,S);t.dispatchEvent(new Event('input',{bubbles:true}));t.dispatchEvent(new Event('change',{bubbles:true}));t.focus();t.scrollIntoView({behavior:'smooth',block:'center'});const o=t.style.outline;t.style.outline='4px solid #00e676';setTimeout(()=>t.style.outline=o,4000);try{const k=td.createElement('div');k.textContent='MKAngel: Treble stake set to £'+S+'. Review and tap Place Bets yourself.';k.style.cssText='position:fixed;top:20px;right:20px;background:#0a7f3f;color:#fff;padding:14px 18px;border-radius:8px;font:14px/1.4 system-ui;z-index:2147483647;max-width:320px;box-shadow:0 6px 20px rgba(0,0,0,.35)';td.body.appendChild(k);setTimeout(()=>k.remove(),5000)}catch(e){}})();
 *
 * ----------------------------------------------------------------------- */
