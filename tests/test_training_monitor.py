import json
from pathlib import Path
from fastapi.testclient import TestClient
from grainmaster.training_monitor import read_metrics, snapshot, create_monitor_app


def test_ignore_incomplete_csv_and_nonfinite_values(tmp_path):
    p=tmp_path/'results.csv'
    p.write_text('epoch,train/seg_loss,metrics/mAP50(M)\n1,1.2,0.5\n2,nan,0.6\n3,0.9,')
    rows=read_metrics(p)
    assert len(rows)==2 and rows[0]['epoch']==1
    assert 'train/seg_loss' not in rows[1]


def test_current_progress_excludes_smoke_and_reports_stale_or_dead(tmp_path):
    state=tmp_path/'artifacts/models/seed_v1';state.mkdir(parents=True)
    (state/'run_status.json').write_text(json.dumps({'stage':'formal_training','pid':123}))
    log=state/'training.log'
    log.write_text('\x1b[K 1/1 10.1G 0.5 1.2 3.1 0.01 5.1 30 1024: 100% 17/17\n'
                   '\x1b[K 2/150 10.1G 0.5 1.2 3.1 0.01 5.1 30 1024: 14% 17/116\n')
    data=snapshot(tmp_path,alive=True,now=log.stat().st_mtime+200)
    assert data['progress']['epoch']==2
    assert data['progress']['batch']==17
    assert data['progress']['batches']==116
    assert data['progress']['percent_of_maximum']<1
    assert data['latest_losses']['seg']==1.2
    assert data['log_stale'] and data['stage']=='formal_training'
    assert snapshot(tmp_path,alive=False)['stage']=='process_missing'
    (state/'run_status.json').write_text(json.dumps({'stage':'training_complete'}))
    assert snapshot(tmp_path,alive=False)['stage']=='training_complete'


def test_endpoint_empty_state_and_preview_path_restriction(tmp_path):
    (tmp_path/'web').mkdir()
    (tmp_path/'web/training.html').write_text('<h1>Monitor</h1>')
    with TestClient(create_monitor_app(tmp_path)) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/training').json()['stage']=='not_started'
        assert client.get('/api/training/preview/weights.pt').status_code==404


def test_windows_run_reads_own_results_and_rejects_path_escape(tmp_path):
    state = tmp_path/'artifacts/models/seed_v1'
    state.mkdir(parents=True)
    folder = tmp_path/'artifacts/models/training/seed_v1_windows'
    folder.mkdir(parents=True)
    (folder/'results.csv').write_text('epoch,train/seg_loss\n6,0.8\n')
    (state/'run_status.json').write_text(json.dumps(dict(stage='formal_training',
        platform='windows', training_name='seed_v1_windows')))
    result = snapshot(tmp_path, alive=True)
    assert result['run_name'] == 'seed_v1_windows'
    assert result['progress']['completed_epochs'] == 6
    (state/'run_status.json').write_text(json.dumps(dict(stage='formal_training',
        training_name='../../outside')))
    assert snapshot(tmp_path)['run_name'] == 'seed_v1_rocm'
