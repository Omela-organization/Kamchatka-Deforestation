import schedule


schedule.every(1).day.do(run)
while True:
    schedule.run_pending()
