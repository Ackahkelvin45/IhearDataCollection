# 1. IF STATEMENT – Check temperature
temperature = float(input("Enter today's temperature in °C: "))
if temperature > 30:
    print("It's a hot day. Stay hydrated!")  # Meteorology example

# 2. IF-ELSE STATEMENT – Check pH level
ph = float(input("Enter the pH value of the solution: "))
if ph < 7:
    print("The solution is acidic.")  # Chemistry example
else:
    print("The solution is neutral or basic.")

# 3. IF-ELIF-ELSE STATEMENT – Classify wind speed
wind_speed = float(input("Enter wind speed in km/h: "))
if wind_speed < 39:
    print("Not a tropical storm.")
elif 39 <= wind_speed < 74:
    print("Tropical storm conditions.")
elif wind_speed >= 74:
    print("Hurricane detected!")
else:
    print("Invalid wind speed.")

# 4. MATCH-CASE STATEMENT – Identify element from symbol
# Note: Requires Python 3.10+
element_symbol = input("Enter a chemical element symbol (e.g., H, O, C): ").upper()

match element_symbol:
    case "H":
        print("Element: Hydrogen")
    case "O":
        print("Element: Oxygen")
    case "C":
        print("Element: Carbon")
    case "N":
        print("Element: Nitrogen")
    case _:
        print("Unknown element")


count = 0
while count < 5:
    print("I love computers")
    count += 1


num = 1
while num <= 5:
    print(num)
    num += 1


N = int(input("Enter number of students: "))
total = 0
count = 0

while count < N:
    score = float(input(f"Enter score for student {count + 1}: "))
    total += score
    count += 1

average = total / N
print("Average score:", average)
