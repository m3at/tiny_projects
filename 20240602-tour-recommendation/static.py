SYSTEM_PROMPT = """\
You are a professional travel agent. 
You generate listings for tours, cultural and physical experiences for your travel agency. You invent activities from plausible ones for the location. You make each listings appealing in their own ways. Not everything is for everybody, but everybody will find something interesting!

For each destination, you start by writing a general description of the place. You then generate 7 or more listings for different activities.
At least one of the listings will be a tour, one a mainly cultural and one mainly physical activity.

You output JSON. 

The format is as follow:

{
  "place": "<city or area's name>",
  "description": "<touristy description about the city or area, in a couple of paragraphs>",
  "listings": [
    {
      "title": "<listing title>",
      "location": "<location>",
      "access": "<how to access it, whether a car is required>",
      "price": "<price in USD>",
      "duration": "<duration of the activity>",
      "requirements": "<when appropriate: fitness level, clothing, equipment, etc>",
      "what to expect": "<description about the activity, a handful of sentences at most>",
      "highlights": "<bullet points highlights with more details, helping customers to get excited about the activity>",
    },
    <... 6 or more other listings ...>
  ]
}
"""


# Prompt:
# Generate a list of 60 touristy destinations across the worlds, some famous, some less so. It can be a city, an area or a general location name. The format is:
# <name>, <country>
#
# Exactly one per line
#
PLACES = [
    "Almaty, Kazakhstan",
    "Amalfi Coast, Italy",
    "Antigua, Guatemala",
    "Auckland, New Zealand",
    "Bagan, Myanmar",
    "Bali, Indonesia",
    "Banff National Park, Canada",
    "Belgrade, Serbia",
    "Bergen, Norway",
    "Bora Bora, French Polynesia",
    "Bruges, Belgium",
    "Budapest, Hungary",
    "Buenos Aires, Argentina",
    "Cairo, Egypt",
    "Cape Town, South Africa",
    "Chiang Mai, Thailand",
    "Cinque Terre, Italy",
    "Cusco, Peru",
    "Dubrovnik, Croatia",
    "Edinburgh, Scotland",
    "Fez, Morocco",
    "Gdansk, Poland",
    "Grand Canyon, USA",
    "Hallstatt, Austria",
    "Havana, Cuba",
    "Helsinki, Finland",
    "Ho Chi Minh City, Vietnam",
    "Istanbul, Turkey",
    "Kyoto, Japan",
    "Lhasa, Tibet",
    "Lisbon, Portugal",
    "Luang Prabang, Laos",
    "Machu Picchu, Peru",
    "Marrakech, Morocco",
    "Medellín, Colombia",
    "Melbourne, Australia",
    "Montevideo, Uruguay",
    "Muscat, Oman",
    "Nairobi, Kenya",
    "Paris, France",
    "Petra, Jordan",
    "Phuket, Thailand",
    "Porto, Portugal",
    "Prague, Czech Republic",
    "Queenstown, New Zealand",
    "Reykjavik, Iceland",
    "Rio de Janeiro, Brazil",
    "Salzburg, Austria",
    "Santiago, Chile",
    "Santorini, Greece",
    "Seoul, South Korea",
    "Seville, Spain",
    "Siem Reap, Cambodia",
    "Sofia, Bulgaria",
    "Sossusvlei, Namibia",
    "Split, Croatia",
    "Sydney, Australia",
    "Tallinn, Estonia",
    "Tulum, Mexico",
    "Ulaanbaatar, Mongolia",
    "Vancouver, Canada",
    "Vienna, Austria",
    "Zanzibar, Tanzania",
    "Zurich, Switzerland",
]
