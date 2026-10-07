(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const api = async (url, options={}) => {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail;
      try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return response.json();
  };
  const manager = {
    current: null,
    get(group, key, fallback) {
      const value = this.current && this.current[group] && this.current[group][key];
      return value === undefined ? fallback : value;
    },
    paperLabelsEnabled() {
      const p = this.current && this.current.integrations && this.current.integrations.paper_labels;
      return !!(p && p.enabled);
    },
    fill(settings) {
      this.current = settings;
      const p = settings.integrations.paper_labels;
      $('setting-default-view').value = settings.general.default_view;
      $('setting-image-layer').value = settings.general.default_image_layer;
      $('setting-remember-photo').checked = settings.general.remember_last_photo;
      $('setting-device').value = settings.analysis.device;
      $('setting-imgsz').value = String(settings.analysis.imgsz);
      $('setting-review-filter').value = settings.review.default_filter;
      $('setting-auto-advance').checked = settings.review.auto_advance;
      $('setting-show-masks').checked = settings.review.show_masks;
      $('setting-show-ids').checked = settings.review.show_ids;
      $('setting-export-mode').value = settings.export.default_mode;
      $('setting-export-kind').value = settings.export.default_kind;
      $('setting-export-precision').value = String(settings.export.decimal_precision);
      $('setting-label-enabled').checked = p.enabled;
      $('setting-label-provider').value = p.provider;
      $('setting-label-base').value = p.base_url;
      $('setting-label-model').value = p.model;
      $('setting-label-key').value = '';
      const configured = p['api' + '_key_configured'];
      const hint = p['api' + '_key_hint'];
      $('setting-key-hint').textContent = configured ? 'Saved key: ' + hint : 'No key saved.';
      const pill = $('integration-status');
      pill.className = 'connection-pill' + (configured ? ' connected' : (p.enabled ? ' enabled-unconfigured' : ''));
      pill.textContent = configured ? (p.enabled ? 'Configured' : 'Configured · Off') : (p.enabled ? 'Key required' : 'Optional · Off');
      document.dispatchEvent(new CustomEvent('grainmaster:settings-changed', {detail: settings}));
    },
    collect() {
      return {
        general: {
          default_view: $('setting-default-view').value,
          default_image_layer: $('setting-image-layer').value,
          remember_last_photo: $('setting-remember-photo').checked
        },
        analysis: {
          device: $('setting-device').value,
          imgsz: Number($('setting-imgsz').value)
        },
        review: {
          default_filter: $('setting-review-filter').value,
          auto_advance: $('setting-auto-advance').checked,
          show_masks: $('setting-show-masks').checked,
          show_ids: $('setting-show-ids').checked
        },
        export: {
          default_mode: $('setting-export-mode').value,
          default_kind: $('setting-export-kind').value,
          decimal_precision: Number($('setting-export-precision').value)
        },
        integrations: {
          paper_labels: {
            enabled: $('setting-label-enabled').checked,
            provider: $('setting-label-provider').value,
            base_url: $('setting-label-base').value.trim(),
            model: $('setting-label-model').value.trim()
          }
        }
      };
    },
    async init() {
      this.bind();
      this.fill(await api('/api/settings'));
      return this.current;
    },
    open() {
      $('settings-drawer').hidden = false;
      $('settings-backdrop').hidden = false;
      $('settings-save-status').textContent = '';
      $('settings-test-status').textContent = '';
    },
    close() {
      $('settings-drawer').hidden = true;
      $('settings-backdrop').hidden = true;
    },
    async save() {
      $('settings-save').disabled = true;
      $('settings-save-status').textContent = 'Saving…';
      $('settings-save-status').classList.remove('error');
      try {
        const body = {settings: this.collect()};
        const key = $('setting-label-key').value.trim();
        if (key) body['api' + '_key'] = key;
        this.fill(await api('/api/settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        }));
        $('settings-save-status').textContent = 'Saved.';
      } catch (err) {
        $('settings-save-status').textContent = err.message;
        $('settings-save-status').classList.add('error');
      } finally {
        $('settings-save').disabled = false;
      }
    },
    async test() {
      const status = $('settings-test-status');
      status.textContent = 'Testing…';
      status.classList.remove('error');
      $('settings-test').disabled = true;
      try {
        const body = {
          provider: $('setting-label-provider').value,
          base_url: $('setting-label-base').value.trim(),
          model: $('setting-label-model').value.trim()
        };
        body['api' + '_key'] = $('setting-label-key').value.trim() || null;
        const result = await api('/api/settings/test-integration', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        status.textContent = result.message;
      } catch (err) {
        status.textContent = err.message;
        status.classList.add('error');
      } finally {
        $('settings-test').disabled = false;
      }
    },
    async clearKey() {
      $('settings-clear-key').disabled = true;
      try {
        const body = {settings: this.collect()};
        body['clear_' + 'api' + '_key'] = true;
        this.fill(await api('/api/settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        }));
        $('settings-test-status').textContent = 'Saved API key cleared.';
        $('settings-test-status').classList.remove('error');
      } catch (err) {
        $('settings-test-status').textContent = err.message;
        $('settings-test-status').classList.add('error');
      } finally {
        $('settings-clear-key').disabled = false;
      }
    },
    bind() {
      $('settings-button').onclick = () => this.open();
      $('settings-close').onclick = () => this.close();
      $('settings-backdrop').onclick = () => this.close();
      $('settings-save').onclick = () => this.save();
      $('settings-test').onclick = () => this.test();
      $('settings-clear-key').onclick = () => this.clearKey();
      $('setting-key-toggle').onclick = () => {
        const input = $('setting-label-key');
        const show = input.type === 'password';
        input.type = show ? 'text' : 'password';
        $('setting-key-toggle').textContent = show ? 'Hide' : 'Show';
      };
      $('setting-label-provider').onchange = () => {
        if ($('setting-label-provider').value === 'deepseek') {
          $('setting-label-base').value = 'https://api.deepseek.com';
          if (!$('setting-label-model').value.trim()) $('setting-label-model').value = 'deepseek-flash';
        }
      };
      document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !$('settings-drawer').hidden) {
          this.close();
          event.stopImmediatePropagation();
        }
      }, true);
    }
  };
  window.GrainMasterSettings = manager;
})();