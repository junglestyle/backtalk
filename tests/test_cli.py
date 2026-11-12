from backtalk.cli import main


def test_main_prints(capsys):
    main()
    assert capsys.readouterr().out
