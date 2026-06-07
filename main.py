from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Keanu Charles Reeves (* 2. September
    1964 in Beirut, Libanon) ist ein kanadischer
    Filmschauspieler, Bassist, Filmregisseur, Autor und
    Filmproduzent. Bekannt wurde er ab 1994 mit Filmen wie
    Speed und den Filmreihen Matrix und John Wick.
    Keanu Reeves wurde in Beirut geboren. Seine Eltern
    Patricia (geborene Taylor, Kostümdesignerin) und Samuel
    Nowlin Reeves Jr. (hawaiisch-chinesischer Herkunft)
    lernten sich in einem Beiruter Nachtclub kennen, in dem
    Patricia als Showgirl arbeitete. Keanu Reeves’ Großmutter
    väterlicherseits ist Chinesin. Er hat eine leibliche
    Schwester und eine Halbschwester mütterlicherseits und
    eine weitere Halbschwester väterlicherseits. Der Name
    Keanu stammt aus der hawaiischen Sprache.
    [3]
    Nach Aufenthalten unter anderem in Australien zog die
    Familie nach New York, wo der Vater die Familie verließ. Seine Mutter Patricia „Patric“ Reeves
    heiratete den Regisseur Paul Aaron; daraufhin zog die Familie nach Toronto, Kanada, wo sich
    Keanus Mutter nach einer nur sechsmonatigen Ehe wieder scheiden ließ. Ihr dritter Ehemann war
    der Promoter Robert Miller, der Vater von Keanus Halbschwester. Später heiratete sie den Friseur
    und Journalisten Jack Bond. 1994 endete auch diese Ehe mit einer Scheidung.
    Im Alter von 15 Jahren besuchte Reeves die Theatre High School, Jesse Ketchum Public School,
    später die North Toronto Secondary und De La Salle College ‘Oaklands’ . Aufgrund seiner
    mangelhaften Disziplin musste er die High School viermal wechseln. In der Saison 1981/82 spielte
    Reeves Eishockey im Team des De La Salle Colleges. Er war Torwart und trug den Spitznamen
    „The Wall“.[4]
    In der High School for the Performing Arts nahm er Schauspielunterricht und wollte fortan
    Schauspieler werden. Im Alter von 17 Jahren brach er die Schule endgültig ab und arbeitete vorerst
    als Schlittschuhschleifer, Koch und Manager einer Pasta-Bude sowie Gärtner, bis er die ersten
    Fernseh- und Theaterrollen erhielt.
    Leben
    Kindheit und Jugend
    07.06.26, 00:03 Keanu Reeves – Wikipedia
    https://de.wikipedia.org/wiki/Keanu_Reeves 1/10
    Keanu Reeves im
    September 2006 in London bei
    der Premiere zu Das Haus am
    See
    Reeves bei der Premiere
    von Pippa Lee auf der
    Berlinale 2009
    In Toronto spielte Reeves am Stadttheater vorwiegend Stücke von
    Shakespeare und sammelte so Bühnenerfahrung. Seine ersten
    Fernseh- und Kinoauftritte hatte er ab 1979 in verschiedenen LowBudget-Produktionen. Sein Filmdebüt gab er in der kanadischen
    Produktion Dream to Believe, bei welcher er auch die für
    Newcomer so begehrte Union Card der Screen Actors Guild
    (genannt SAG, eine Schauspieler-Gewerkschaft) erwarb, ohne die
    kaum jemand im Showgeschäft zu wirklich guten Rollen kam. Mit
    20 Jahren riet ihm sein damaliger Manager zu einem
    Künstlernamen. Daraufhin überlegte Reeves, den Künstlernamen
    Chuck Spadina zu verwenden, entschied sich letztlich aber
    dagegen.[5] 1986 verließ er Kanada mit 3000 US-Dollar, einem
    alten Volvo und der Adresse seines Stiefvaters Paul Aaron.
    Zunächst spielte er die Hauptrolle in Der Prinz von Pennsylvania
    als Sohn von Fred Ward. Mit der Rolle des trotteligen Teenagers
    Ted in Bill & Teds verrückte Reise durch die Zeit gelang ihm der
    Durchbruch in Hollywood. Weltbekannt wurde er aber erst 1994 als
    wagemutiger Polizist im Blockbuster Speed, der auch seiner
    Filmpartnerin Sandra Bullock den Durchbruch bescherte. Fortan
    erhielt er Gagen in Millionenhöhe, blieb aber vielen Kritikern ein
    Dorn im Auge, die seine Schauspielkunst als hölzern und
    ausdruckslos bezeichneten. Hollywood allerdings machte ihm ein
    Filmangebot nach dem anderen. Reeves konnte es sich 1997 leisten,
    das Angebot einer 11-Millionen-Dollar-Gage für Speed 2
    auszuschlagen und stattdessen mit seiner Band Dogstar auf Tour
    zu gehen und die Titelrolle in Hamlet am Manitoba Theater Center,
    Winnipeg, zu spielen.
    Die Rolle des Computerhackers Neo in der Matrix-Trilogie (vier
    Oscars) machte ihn 1999 zu einem der bestbezahlten
    Hollywoodstars. Der ursprünglich für diese Rolle favorisierte Will
    Smith hatte sich für den Film Wild Wild West entschieden.[6] Die
    negative Kritik an Reeves wurde seitdem spärlicher. Mit Rollen wie
    der des brutalen Rednecks in The Gift oder des verliebten Arztes in
    Was das Herz begehrt erntete er viel Lob.
    Bei den 62. Internationalen Filmfestspielen 2012 in Berlin stellte Reeves seinen in Kritikerkreisen
    viel beachteten Film Side by Side vor.[7] Die Dokumentation beschäftigt sich mit der Frage, welche
    Veränderungen sich beim Filmemachen durch die digitale Technik ergeben. Im deutschsprachigen
    Raum existiert eine gekürzte Fassung dieser Dokumentation unter dem Titel Kino reloaded, die
    Ende 2012 beim Sender ServusTV ihre Fernsehpremiere hatte. 2013 gab Reeves mit dem Film
    Man of Tai Chi sein Regiedebüt. Der Film wurde in China gedreht; Reeves übernahm die Rolle des
    Antagonisten.
    2014 erlebte er mit der Rolle des Auftragskillers John Wick in dem gleichnamigen Film erneut
    einen Durchbruch in seiner Karriere. 2015 spielte Reeves die Hauptrolle in dem ErotikHorrorthriller Knock Knock, 2016 spielte er eine Rolle in dem Film The Neon Demon. 2017 spielte
    er erneut die Rolle des John Wick in der Fortsetzung John Wick: Kapitel 2, auch 2019 verkörperte
    Filmkarriere
    07.06.26, 00:03 Keanu Reeves – Wikipedia
    https://de.wikipedia.org/wiki/Keanu_Reeves 2/10
    er erneut John Wick im dritten Teil John Wick: Kapitel 3. In John Wick: Kapitel 4, der am 23.
    März 2023 in die deutschen Kinos kam, spielt er wieder die Hauptrolle als John Wick. Die
    Deutschlandpremiere, bei der Keanu Reeves persönlich anwesend war, fand am 8. März in Berlin
    statt.[8]
    Im selben Jahr spielte er eine Rolle im Netflix-Film Always Be My Maybe, wofür er von
    der Presse und den Fans gefeiert wurde. In From the World of John Wick: Ballerina aus dem Jahr
    2025 spielte er erneut die Rolle des John Wick.
    Reeves spricht eine Rolle in dem Pixar-Animationsfilm A Toy Story: Alles hört auf kein
    Kommando. Im Juni 2019 trat er auf der E3 auf, um für das Spiel Cyberpunk 2077 zu werben, in
    dem er eine wichtige Rolle spielt. Für seinen Auftritt wurde Reeves abermals von seinen Fans und
    der Presse gefeiert. Das Jahr 2019 wird von vielen auch als „Das Jahr des Keanu Reeves“
    bezeichnet.[9] Reeves wird in der deutschen Synchronisation seit etwa 1993 von Benjamin Völz
    gesprochen. Zuvor kamen auch Frank Schaff, Nicolas Böll, Pascal Breuer, Udo Schenk oder
    Joachim Tennstedt zum Zuge
    """

    summary_template = """
    given the information {information} about a person i want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5.4-mini")
    # llm = ChatOllama(temperature=0, model="gemma4:e2b")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
