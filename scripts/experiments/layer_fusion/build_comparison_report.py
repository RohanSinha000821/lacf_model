"""Build a saved-score-only full-F1 report; never runs a detector or new bootstrap."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from audio_deepfake_detection.metrics import compute_auroc, evaluate_transfer, read_score_csv
from audio_deepfake_detection.protocol import canonical_score_labels, checkpoint_digest

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'outputs/experiments/layer_fusion/reports/f1_frozen_fusion'
DATA = Path('/mnt/drive/audio-deepfake-cache')
PDF = OUT/'Frozen_Audio_Fusion_F1_Comparison.pdf'
SOURCES = ('asv2019','asv5','cfad')
MODELS = (
    ('anchor', 'Anchor-Relation LACF', 'Anchor LACF',
     'outputs/lacf10s_optimized_20261009/f1/1234',
     'outputs/lacf10s_optimized_20261009/f1/1234', 661479),
    ('moe', 'LACF Layer-MoE', 'Layer-MoE',
     'outputs/experiments/layer_fusion/lacf_layer_moe/f1/1234',
     'outputs/experiments/layer_fusion/evaluations/lacf_layer_moe/f1/1234', 967338),
    ('transformer', 'Audio Token Transformer Fusion', 'Transformer Fusion',
     'outputs/experiments/token_fusion/full_source_f1_20261029T231407+0100/TF2/f1/1234',
     'outputs/experiments/token_fusion/eval_f1_speechfake_20261030T131151+0100/TF2/f1/1234', 4907745),
)


def read(path):
    return json.loads(path.read_text())


def close(a,b):
    assert np.isclose(a,b,rtol=0,atol=1e-12),(a,b)


def collect():
    canonical={d:canonical_score_labels(d,'dev',DATA) for d in SOURCES}
    canonical['speechfake']=canonical_score_labels('speechfake','test',DATA)
    results=[]
    for key,name,short,training,evaluation,parameters in MODELS:
        train,run=ROOT/training,ROOT/evaluation
        report=read(run/'metrics.json');complete=read(train/'training_complete.json');config=read(train/'config.json')
        assert report['fold']=='f1' and report['seed']==1234 and report['target_domain']=='speechfake'
        assert report['sources']==list(SOURCES) and complete['selected_checkpoint_reloaded_and_source_verified']
        assert checkpoint_digest(train/'best.pt')==report['checkpoint_sha256']==complete['best_checkpoint_sha256']
        assert report['checkpoint_epoch']==complete['best_epoch']
        assert report['confidence_intervals']['resamples']==1000 and report['confidence_intervals']['seed']==2026
        assert report['confidence_intervals']['class_stratified']
        assert report['confidence_intervals']['includes_training_or_calibration_uncertainty'] is False
        source={}
        for d in SOURCES:
            path=run/f'source_dev_{d}.csv'
            assert checkpoint_digest(path)==report['score_sha256'][path.name]
            source[d]=read_score_csv(path,d,expected_labels=canonical[d])
        target=run/'target_scores.csv'
        assert checkpoint_digest(target)==report['score_sha256'][target.name]
        labels,scores=read_score_csv(target,'speechfake',expected_labels=canonical['speechfake'])
        reproduced=evaluate_transfer(source,labels,scores,bootstrap_resamples=0)
        for metric in ('eer','auroc'):close(reproduced['target'][metric],report['target'][metric])
        close(reproduced['source_dev']['macro_eer'],report['source_dev']['macro_eer'])
        for d in SOURCES:close(reproduced['source_dev']['eer_by_domain'][d],report['source_dev']['eer_by_domain'][d])
        for branch in ('strict_raw_transfer','unlabeled_zscore_transfer'):
            for rate in ('1%','5%','10%'):
                close(reproduced[branch]['thresholds'][rate],report[branch]['thresholds'][rate])
                for metric in ('apcer','bpcer','acer'):
                    close(reproduced[branch]['target_operating_points'][rate][metric],report[branch]['target_operating_points'][rate][metric])
        ci=report['confidence_intervals']['intervals']
        assert all(len(bounds)==2 and bounds[0]<=bounds[1] for bounds in ci.values())
        for metric in ('eer','auroc'):
            assert ci[metric][0]<=report['target'][metric]<=ci[metric][1]
        source_auc={d:compute_auroc(*source[d]) for d in SOURCES}
        results.append({'key':key,'name':name,'short':short,'parameters':parameters,'training':training,'evaluation':evaluation,
                        'report':report,'completion':complete,'source_auroc':source_auc,
                        'source_macro_auroc':float(np.mean(list(source_auc.values()))),
                        'metrics_sha256':checkpoint_digest(run/'metrics.json'),
                        'config_sha256':checkpoint_digest(train/'config.json'),
                        'training_input':config.get('training_views',config.get('input_policy'))})
    assert all(x['report']['target']['n_bona_fide']==37242 and x['report']['target']['n_spoof']==309065 for x in results)
    return results


FONT_ROOT=Path('/usr/share/fonts/truetype/dejavu')
for name,file in (('Report','DejaVuSans.ttf'),('ReportBold','DejaVuSans-Bold.ttf'),('ReportItalic','DejaVuSans-Oblique.ttf')):
    pdfmetrics.registerFont(TTFont(name,str(FONT_ROOT/file)))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBold')
NAVY=colors.HexColor('#15314B');TEAL=colors.HexColor('#176D79');INK=colors.HexColor('#243447')
PALE=colors.HexColor('#EFF5F8');LINE=colors.HexColor('#D6E1E8');MUTED=colors.HexColor('#536578')
STYLE={
 'title':ParagraphStyle('title',fontName='ReportBold',fontSize=22,leading=27,textColor=NAVY,spaceAfter=8),
 'h1':ParagraphStyle('h1',fontName='ReportBold',fontSize=15,leading=20,textColor=NAVY,spaceAfter=10),
 'h2':ParagraphStyle('h2',fontName='ReportBold',fontSize=11,leading=15,textColor=TEAL,spaceBefore=9,spaceAfter=5),
 'body':ParagraphStyle('body',fontName='Report',fontSize=9,leading=13,textColor=INK,spaceAfter=7),
 'small':ParagraphStyle('small',fontName='Report',fontSize=7.8,leading=11,textColor=MUTED,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Report',fontSize=8,leading=11,textColor=INK),
 'cellsmall':ParagraphStyle('cellsmall',fontName='Report',fontSize=7.1,leading=10,textColor=INK),
 'number':ParagraphStyle('number',fontName='Report',fontSize=8.3,leading=11,alignment=TA_CENTER,textColor=INK),
 'head':ParagraphStyle('head',fontName='ReportBold',fontSize=7.8,leading=11,textColor=colors.white),
}


def p(text,style='body'):
    return Paragraph(text,STYLE[style])


def table(rows,widths,*,small=False):
    content=[]
    for index,row in enumerate(rows):
        content.append([value if isinstance(value,Paragraph) else p(str(value),'head' if index==0 else ('cellsmall' if small else 'cell')) for value in row])
    value=Table(content,colWidths=widths,hAlign='LEFT',repeatRows=1)
    value.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),.5,NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
        ('LINEBELOW',(0,1),(-1,-1),.35,LINE),
    ]))
    return value


def interval(value,bounds,*,percent=True,digits=3):
    scale=100 if percent else 1
    return p(f'<b>{value*scale:.{digits}f}</b><br/><font size="7">[{bounds[0]*scale:.{digits}f}, {bounds[1]*scale:.{digits}f}]</font>','number')


def footer(canvas,doc):
    width,height=A4
    canvas.saveState();canvas.setStrokeColor(LINE);canvas.line(42,38,width-42,38)
    canvas.setFont('Report',7);canvas.setFillColor(MUTED)
    canvas.drawString(42,25,'F1 · seed 1234 · frozen audio encoders · saved results only')
    canvas.drawRightString(width-42,25,f'{doc.page}')
    canvas.restoreState()


def write_pdf(results):
    doc=BaseDocTemplate(str(PDF),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=50,
        title='Frozen Audio Fusion: Full-Fold F1 Comparison',author='Rohan Sinha',subject='LACF, layer mixture of experts and audio-token Transformer fusion; matched seed1234 F1 results')
    doc.addPageTemplates(PageTemplate(id='report',frames=Frame(42,50,A4[0]-84,A4[1]-88,leftPadding=0,bottomPadding=0,rightPadding=0,topPadding=0),onPage=footer))
    story=[];width=A4[0]-84
    story += [p('Frozen Audio Fusion','title'),p('Full-fold F1 comparison · 10 November 2026','h1'),
       p('Anchor-Relation LACF · LACF Layer-MoE · Audio Token Transformer Fusion','small'),
       p('<b>Question:</b> Does direct Transformer fusion provide a stronger direction than the original language-anchor detector and its layer-expert extension?'),
       p('<b>Evidence:</b> Transformer Fusion has the strongest ranking on this completed F1 test: lower EER and higher AUROC. Its source-derived thresholds transfer less reliably, so it is not the best system at every operating point.','body'),
       p('Matched comparison','h2'),
       p('All three use seed <b>1234</b>, frozen WavLM Base+ and unfused CLAP audio, the same three F1 sources (ASV2019 LA, ASV5 and CFAD), and SpeechFake test as the target. Source train/dev availability: <b>246,137 / 180,194</b>. Target: <b>346,307</b> recordings (37,242 bona fide; 309,065 spoof). Target scoring uses a centered common interval up to 10 seconds, with shorter audio preserved. No new inference or bootstrap was run for this report.'),
       p('What the model names mean','h2')]
    methods=[['Model','Trainable parameters','Readout and objective'],
      ['<b>Anchor-Relation LACF</b>', '661,479','Last WavLM layer → 512-D adapter; two audio branches → eight text-anchor probabilities → 35-D relations. Detection BCE + 0.25 semantic BCE + 0.10 bona-fide JS loss.'],
      ['<b>LACF Layer-MoE</b>','967,338','All 13 WavLM layers → three layer-group residual experts → audio-dependent mixture → original adapter/35-D relation head. Detection BCE only; both audio branches and anchors retained.'],
      ['<b>Audio Token Transformer Fusion</b>','4,907,745','13 WavLM layer tokens + one CLAP audio token + a classification token; two 512-D Transformer blocks. A 512→64→16→1 classifier uses the classification-token output. Detection BCE only; no text anchors.']]
    story += [table(methods,[118,83,width-201],small=True),p('Source-selected checkpoints','h2')]
    rows=[['Model','Best / final epoch','Macro source EER (%)','Macro source AUROC']]
    for x in results:
        rows.append([x['short'],f"{x['completion']['best_epoch']} / {x['completion']['final_epoch']}",f"{100*x['report']['source_dev']['macro_eer']:.4f}",f"{x['source_macro_auroc']:.6f}"])
    story += [table(rows,[145,90,135,width-370]),p('Held-out SpeechFake ranking','h2')]
    rows=[['Model','EER (%) ↓ [95% CI]','AUROC ↑ [95% CI]']]
    for x in results:
        r=x['report'];ci=r['confidence_intervals']['intervals'];rows.append([x['short'],interval(r['target']['eer'],ci['eer'],digits=2),interval(r['target']['auroc'],ci['auroc'],percent=False,digits=4)])
    story += [table(rows,[145,183,width-328]),p('Intervals describe recording-level resampling conditional on the fixed checkpoint/calibration. They are not training-seed uncertainty or a paired model-difference test.','small'),PageBreak()]

    for branch,title,description in (
        ('strict_raw_transfer','Strict raw threshold transfer','Each model’s thresholds are calibrated once on pooled source-development spoof scores, then transferred unchanged to SpeechFake. No target adjustment is used.'),
        ('unlabeled_zscore_transfer','Transductive z-score threshold transfer','Each source domain is standardized using its own score mean/standard deviation before pooled source calibration. Target scores use the complete unlabeled target mean/standard deviation. This uses target-distribution information and is reported separately from strict transfer.')):
        story += [p(title,'h1'),p(description),p('<b>Every error is a percentage.</b> Each cell gives the point estimate with its saved 95% bootstrap interval below. “Source APCER” labels the nominal source calibration point; it does not require target APCER to equal that percentage. Lower errors are better.'),Spacer(1,5)]
        rows=[['Model','Source<br/>APCER','Threshold','Target APCER<br/>(spoofs missed)','Target BPCER<br/>(genuine rejected)','Target ACER<br/>(mean error)']]
        prefix='strict_raw' if branch=='strict_raw_transfer' else 'unlabeled_zscore'
        for x in results:
            r=x['report'];ci=r['confidence_intervals']['intervals']
            for rate in ('1%','5%','10%'):
                op=r[branch]['target_operating_points'][rate]
                rows.append([x['short'],rate,f"{r[branch]['thresholds'][rate]:.4f}",*[interval(op[m],ci[f'{prefix}.{rate}.{m}']) for m in ('apcer','bpcer','acer')]])
        story += [table(rows,[104,45,60,(width-209)/3,(width-209)/3,(width-209)/3],small=True),Spacer(1,12)]
        if branch=='strict_raw_transfer':
            story += [p('What this changes in the interpretation','h2'),
             p('<b>Transformer Fusion is worse on raw ACER at all three transferred thresholds.</b> At the source 5% setting, its target ACER is <b>38.761%</b>, versus <b>31.653%</b> for Layer-MoE and <b>23.190%</b> for Anchor LACF. Its high APCER means many spoofs receive scores below the source threshold.'),
             p('At the source 1% setting, Layer-MoE has the lowest ACER (22.606%). Anchor LACF has the lowest raw ACER at the source 5% and 10% settings. These are fixed, prespecified operating points; no favorable threshold is selected from the target.'),
             p('<b>Definitions:</b> APCER = spoof accepted as bona fide; BPCER = bona-fide recording rejected as spoof; ACER = (APCER + BPCER)/2. Higher detector scores mean spoof, with spoof predicted when score ≥ threshold.'),
             p('Raw thresholds are model-specific scalar spoof-logit values. Their numerical magnitudes are not a common confidence scale. Evaluation used unrounded saved thresholds.','small')]
        else:
            story += [p('Normalization helps some settings, not all','h2'),
             p('Transformer Fusion has the lowest transductive ACER at the source 1% setting (<b>18.000%</b>); Layer-MoE is close at <b>18.260%</b>. No paired significance test was performed for this difference. Layer-MoE has the lowest transductive ACER at the 5% and 10% settings (<b>18.947%</b> and <b>20.080%</b>).'),
             p('Anchor LACF at the 1% setting classifies every target as spoof: APCER 0%, BPCER 100%, ACER 50%. Its zero-width conditional interval describes these fixed observed scores; it does not imply certainty for future recordings or retraining.'),
             p('Positive z-score scaling preserves score ordering, so EER/AUROC remain the ranking values on page 1. Z-score thresholds are dimensionless. Neither this branch nor a source operating point is selected using target labels.','small')]
        story.append(PageBreak())

    story += [p('Source results, evidence limits and decision','h1'),p('Per-source development metrics','h2'),p('Each cell is EER (%) / AUROC. Macro rows weight the three source domains equally; they are not pooled across recordings. No source-bootstrap confidence intervals were calculated.','small')]
    rows=[['Source domain','Anchor LACF','Layer-MoE','Transformer Fusion']]
    for d,label in [('asv2019','ASV2019 LA'),('asv5','ASV5'),('cfad','CFAD')]:
        rows.append([label,*[f"{100*x['report']['source_dev']['eer_by_domain'][d]:.4f} / {x['source_auroc'][d]:.6f}" for x in results]])
    rows.append(['Equal-domain macro',*[f"{100*x['report']['source_dev']['macro_eer']:.4f} / {x['source_macro_auroc']:.6f}" for x in results]])
    story += [table(rows,[112,(width-112)/3,(width-112)/3,(width-112)/3],small=True),
      p('What can we conclude?','h2'),
      p('<b>Supported:</b> on this full-F1, seed-1234 comparison, direct audio-token Transformer fusion provides the best class ranking. Its EER is <b>7.927 percentage points lower</b> than Anchor LACF and <b>3.741 points lower</b> than Layer-MoE; its AUROC is also higher. Layer-MoE improves EER over Anchor LACF by <b>4.186 points</b>.'),
      p('<b>Qualified direction:</b> the Transformer-fusion architecture is promising for representation fusion. It has not established a universally better detector: source-threshold transfer remains weak, and Layer-MoE or Anchor LACF wins several fixed operating points. These results do not justify selecting a new threshold, branch or architecture using SpeechFake.'),
      p('What this comparison does not isolate','h2'),
      p('Both audio encoders are frozen in all three completed systems, but the trained heads differ substantially in capacity (0.661M, 0.967M and 4.908M parameters). Transformer Fusion preserves layer/audio embeddings for attention, removes the anchor/relation bottleneck, and omits both auxiliary losses. The results do not prove that attention alone, the text concepts, or any one loss caused the differences.'),
      p('Anchor LACF trained on newly sampled random crops with padded WavLM batches. The cached models use a centered view plus two fixed random views and native unpadded encoder extraction. They therefore compare complete systems, not one perfectly isolated architectural change. The same frozen target-feature cache is reused for Layer-MoE and Transformer Fusion, with recorded BF16 extraction rounding retained.'),
      p('Scope and uncertainty','h2'),
      p('This is one exploratory fold/seed after earlier target exposure. Checkpoints and calibration were selected on source development only. Saved CIs use 1,000 class-stratified target resamples (seed 2026), with checkpoint, thresholds and normalization fixed. They exclude training/calibration uncertainty and speaker/generator clustering. Separate intervals are not a paired significance test.'),
      p('Provenance and verification','h2')]
    rows=[['Model','Selected checkpoint SHA-256 prefix','Saved metrics SHA-256 prefix']]
    for x in results:rows.append([x['short'],x['report']['checkpoint_sha256'][:16],x['metrics_sha256'][:16]])
    story += [table(rows,[145,183,width-328],small=True),
      p('Point metrics and thresholds reproduce from hash-verified score CSVs with matching canonical IDs/labels. Saved CIs were preserved; no inference, bootstrap, tuning or job changes were run. Full paths/hashes/unrounded values are in comparison_data.json and comparison_metrics.csv.','small'),
      p('Lineage: Anchor LACF = lacf10s_optimized_20261009; Layer-MoE = diagnostic C4 fitted on full F1; Transformer Fusion = frozen token-fusion condition TF2. Ongoing encoder-adaptation runs are excluded.','small')]
    doc.build(story)


def companions(results):
    save={'created_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(),'scope':'FullF1 seed1234; all three frozen audio encoder systems',
          'verification':'Canonical membership/labels and saved CSV/checkpoint hashes verified; all point metrics/thresholds reproduced, saved CIs unchanged',
          'new_inference':False,'new_bootstrap':False,'models':results}
    (OUT/'comparison_data.json').write_text(json.dumps(save,indent=2)+'\n')
    with (OUT/'comparison_metrics.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['model','seed','partition','domain_or_branch','source_apcer','metric','estimate','ci_low','ci_high','unit'])
        for x in results:
            r=x['report'];ci=r['confidence_intervals']['intervals']
            for metric in ('eer','auroc'):
                writer.writerow([x['name'],1234,'target','speechfake','',metric,r['target'][metric],*ci[metric],'fraction'])
            for branch,prefix in (('strict_raw_transfer','strict_raw'),('unlabeled_zscore_transfer','unlabeled_zscore')):
                for rate in ('1%','5%','10%'):
                    writer.writerow([x['name'],1234,'target',branch,rate,'threshold',r[branch]['thresholds'][rate],'','','logit' if prefix=='strict_raw' else 'zscore'])
                    for metric in ('apcer','bpcer','acer'):
                        writer.writerow([x['name'],1234,'target',branch,rate,metric,r[branch]['target_operating_points'][rate][metric],*ci[f'{prefix}.{rate}.{metric}'],'fraction'])
            for d in SOURCES:
                writer.writerow([x['name'],1234,'source_dev',d,'','eer',r['source_dev']['eer_by_domain'][d],'','','fraction'])
                writer.writerow([x['name'],1234,'source_dev',d,'','auroc',x['source_auroc'][d],'','','fraction'])
            writer.writerow([x['name'],1234,'source_dev','macro','','eer',r['source_dev']['macro_eer'],'','','fraction'])
            writer.writerow([x['name'],1234,'source_dev','macro','','auroc',x['source_macro_auroc'],'','','fraction'])


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if PDF.exists():raise FileExistsError('Preserve existing comparison PDF')
    results=collect()
    companions(results)
    write_pdf(results)
    print(PDF)
    print('Verified all saved point metrics/source thresholds, checkpoint/score hashes and canonical IDs/labels; no inference or new bootstrap.')
