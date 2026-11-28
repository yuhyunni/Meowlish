import json, pathlib, subprocess, numpy as np, soundfile as sf
from kokoro import KPipeline
root=pathlib.Path(__file__).resolve().parent.parent
out=root/'kokoro-audio';out.mkdir(exist_ok=True)
manifest=json.loads((root/'kokoro-tools'/'manifest.json').read_text())
pipeline=KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M')
for text,item in manifest.items():
    target=out/item['file']
    if target.exists() and target.stat().st_size>1000: continue
    chunks=[np.asarray(audio,dtype=np.float32) for _,_,audio in pipeline(text,voice=item['voice'],speed=1.0)]
    if not chunks: raise RuntimeError(f'No audio: {text}')
    wav=target.with_suffix('.wav'); sf.write(wav,np.concatenate(chunks),24000)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(wav),'-codec:a','libmp3lame','-b:a','64k',str(target)],check=True)
    wav.unlink(); print(target.name,flush=True)
print(f'Generated or reused {len(manifest)} files')
