describe("test1", () => {
    test("test222", () => {
        expect(
            (() => {
                return 1;
            })()
        ).toBe(1);
    });
});
