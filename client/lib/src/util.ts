export const createDummyMediaStream = (audioContext: AudioContext) => {
    const dummyOutputNode = audioContext.createMediaStreamDestination();

    const gainNode = audioContext.createGain();
    gainNode.gain.value = 0.0;
    gainNode.connect(dummyOutputNode);
    const oscillatorNode = audioContext.createOscillator();
    oscillatorNode.frequency.value = 440;
    oscillatorNode.connect(gainNode);
    oscillatorNode.start();
    return dummyOutputNode.stream;
};
