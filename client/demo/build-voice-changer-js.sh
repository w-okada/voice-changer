cd ~/git-work/voice-changer-js/lib/ ; npm run build:dev; cd -
rm -rf node_modules/@dannadori/voice-changer-js
mkdir -p node_modules/@dannadori/voice-changer-js/dist
cp -r ~/git-work/voice-changer-js/lib/package.json node_modules/@dannadori/voice-changer-js/
cp -r ~/git-work/voice-changer-js/lib/dist node_modules/@dannadori/voice-changer-js/
